"""
Implement the ES-MDA algorithms.

@author: acollet
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from pyesmda.base import ESMDABase
from pyesmda.inversion import ESMDAInversionType
from pyesmda.localization import LocalizationStrategy, NoLocalization
from pyesmda.utils import NDArrayFloat

# pylint: disable=C0103 # Does not conform to snake_case naming style


class ESMDA(ESMDABase):
    r"""
    Ensemble Smoother with Multiple Data Assimilation.

    Implement the ES-MDA as proposed by  Emerick, A. A. and A. C. Reynolds
    :cite:p:`emerickEnsembleSmootherMultiple2013,
    emerickHistoryMatchingProductionSeismic2013`.

    Attributes
    ----------
    d_dim : int
        Number of observation values :math:`N_{obs}`, and consequently of
        predicted values.
    obs : NDArrayFloat
        Obsevrations vector with dimensions (:math:`N_{obs}`).
    cov_obs: NDArrayFloat
        Covariance matrix of observed data measurement errors with dimensions
        (:math:`N_{obs}`, :math:`N_{obs}`). Also denoted :math:`R`.
    d_obs_uc: NDArrayFloat
        Vectors of pertubed observations with dimension
        (:math:`N_{obs}`, :math:`N_{e}`).
    d_pred: NDArrayFloat
        Vectors of predicted values (one for each ensemble member)
        with dimensions (:math:`N_{obs}`, :math:`N_{e}`).
    d_history: List[NDArrayFloat]
        List of vectors of predicted values obtained at each assimilation step.
    m_prior:
        Vectors of parameter values (one vector for each ensemble member) used in the
        last assimilation step. Dimensions are (:math:`N_{m}`, :math:`N_{e}`).
    m_bounds : NDArrayFloat
        Lower and upper bounds for the :math:`N_{m}` parameter values.
        Expected dimensions are (:math:`N_{m}`, 2) with lower bounds on the first
        column and upper on the second one.
    m_history: List[NDArrayFloat]
        List of successive `m_prior`.
    cov_md: NDArrayFloat
        Cross-covariance matrix between the forecast state vector and predicted data.
        Dimensions are (:math:`N_{m}, N_{obs}`).
    cov_obs: csr_matrix
        Autocovariance matrix of predicted data.
        Dimensions are (:math:`N_{obs}, N_{obs}`).
    cov_mm: NDArrayFloat
        Autocovariance matrix of estimated parameters.
        Dimensions are (:math:`N_{m}, N_{m}`).
    forward_model: callable
        Function calling the non-linear observation model (forward model)
        for all ensemble members and returning the predicted data for
        each ensemble member.
    forward_model_args: Tuple[Any]
        Additional args for the callable forward_model.
    forward_model_kwargs: Dict[str, Any]
        Additional kwargs for the callable forward_model.
    n_assimilations : int
        Number of data assimilations (:math:`N_{a}`).
    cov_obs_inflation_factors : List[float]
        List of multiplication factor used to inflate the covariance matrix of the
        measurement errors.
    cov_mm_inflation_factor: float
        Factor used to inflate the initial ensemble around its mean.
        See :cite:p:`andersonExploringNeedLocalization2007`.
        The default is 1.0 i.e., no inflation.
    C_DD_localization: LocalizationStrategy
        Localization operator :math:`\rho_{DD}` applied to the predictions
        empirical auto-covariance matrices. Expected dimensions of the operator are
        (:math:`N_{obs}`, :math:`N_{obs}`). It can be fixed (defined correlation
        matrix used for all iterations) or adaptive and even user defined.
        See implementations of :class:`LocalizationStrategy`.
    C_MD_localization : Optional[csr_matrix]
        Localization operator :math:`\rho_{DD}` applied to the parameters-predictions
        empirical corss-covariance matrices. Expected dimensions of the operator are
        (:math:`N_{m}`, :math:`N_{obs}`). It can be fixed (defined correlation
        matrix used for all iterations) or adaptive and even user defined.
        See implementations of :class:`LocalizationStrategy`.
    save_ensembles_history: bool
        Whether to save the history predictions and parameters over the assimilations.
    rng: np.random.Generator
        The random number generator used in the predictions perturbation step.
    is_forecast_for_last_assimilation: bool
        Whether to compute the predictions for the ensemble obtained at the
        last assimilation step.
    batch_size: int
        Number of parameters that are assimilated at once. This option is
        available to overcome memory limitations when the number of parameters is
        large. In that case, the size of the covariance matrices tends to explode
        and the update step must be performed by chunks of parameters.
    is_parallel_analyse_step: bool, optional
        Whether to use parallel computing for the analyse step if the number of
        batch is above one. The default is True.
    n_batches: int
        Number of batches required during the update step.
    truncation: float
        A value in the range ]0, 1], used to determine the number of
        significant singular values kept when using svd for the inversion
        of $(C_{dd} + \alpha C_{d})$: Only the largest singular values are kept,
        corresponding to this fraction of the sum of the nonzero singular values.
        The goal of truncation is to deal with smaller matrices (dimensionality
        reduction), easier to inverse.
    logger: Optional[logging.Logger]
        Optional :class:`logging.Logger` instance used for event logging.

    """

    # pylint: disable=R0902 # Too many instance attributes
    __slots__: List[str] = [
        "_cov_obs_inflation_factors",
    ]

    def __init__(
        self,
        obs: NDArrayFloat,
        m_init: NDArrayFloat,
        cov_obs: NDArrayFloat,
        forward_model: Callable[..., NDArrayFloat],
        forward_model_args: Sequence[Any] = (),
        forward_model_kwargs: Optional[Dict[str, Any]] = None,
        n_assimilations: int = 4,
        inversion_type: Union[ESMDAInversionType, str] = ESMDAInversionType.NAIVE,
        cov_obs_inflation_factors: Optional[Sequence[float]] = None,
        cov_mm_inflation_factor: float = 1.0,
        C_DD_localization: LocalizationStrategy = NoLocalization(),
        C_MD_localization: LocalizationStrategy = NoLocalization(),
        m_bounds: Optional[NDArrayFloat] = None,
        save_ensembles_history: bool = False,
        seed: Optional[int] = None,
        is_forecast_for_last_assimilation: bool = True,
        random_state: Optional[
            Union[int, np.random.Generator, np.random.RandomState]
        ] = 198873,
        batch_size: int = 5000,
        is_parallel_analyse_step: bool = True,
        truncation: float = 0.99,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        # pylint: disable=R0913 # Too many arguments
        # pylint: disable=R0914 # Too many local variables
        r"""Construct the instance.

        Parameters
        ----------
        obs : NDArrayFloat
            Obsevrations vector with dimension :math:`N_{obs}`.
        m_init : NDArrayFloat
            Initial ensemble of parameters vector with dimensions
            (:math:`N_{m}`, :math:`N_{e}`).
        cov_obs: NDArrayFloat
            Covariance matrix of observed data measurement errors with dimensions
            (:math:`N_{obs}`, :math:`N_{obs}`). Also denoted :math:`R`.
            It can be a numpy array or a sparse matrix (scipy.linalg).
        forward_model: callable
            Function calling the non-linear observation model (forward model)
            for all ensemble members and returning the predicted data for
            each ensemble member.
        forward_model_args: Optional[Tuple[Any]]
            Additional args for the callable forward_model. The default is None.
        forward_model_kwargs: Optional[Dict[str, Any]]
            Additional kwargs for the callable forward_model. The default is None.
        n_assimilations : int, optional
            Number of data assimilations (:math:`N_{a}`). The default is 4.
        cov_obs_inflation_factors : Optional[Sequence[float]]
            Multiplication factor used to inflate the covariance matrix of the
            measurement errors.
            Must match the number of data assimilations (:math:`N_{a}`).
            The default is None.
        cov_mm_inflation_factor: float
            Factor used to inflate the initial ensemble around its mean.
            See :cite:p:`andersonExploringNeedLocalization2007`.
            The default is 1.0 i.e., no inflation.
        C_DD_localization: LocalizationStrategy
            Localization operator :math:`\rho_{DD}` applied to the predictions
            empirical auto-covariance matrices. Expected dimensions of the operator are
            (:math:`N_{obs}`, :math:`N_{obs}`). It can be fixed (defined correlation
            matrix used for all iterations) or adaptive and even user defined.
            See implementations of :class:`LocalizationStrategy`.
        C_MD_localization : Optional[csr_matrix]
            Localization operator :math:`\rho_{DD}` applied to the
            parameters-predictions empirical corss-covariance matrices.
            Expected dimensions of the operator are
            (:math:`N_{m}`, :math:`N_{obs}`). It can be fixed (defined correlation
            matrix used for all iterations) or adaptive and even user defined.
            See implementations of :class:`LocalizationStrategy`.
        m_bounds : Optional[NDArrayFloat], optional
            Lower and upper bounds for the :math:`N_{m}` parameter values.
            Expected dimensions are (:math:`N_{m}`, 2) with lower bounds on the first
            column and upper on the second one. The default is None.
        save_ensembles_history: bool, optional
            Whether to save the history predictions and parameters over
            the assimilations. The default is False.
        seed: Optional[int]
            .. deprecated:: 0.4.2
                Since 0.4.2, you can use the parameter `random_state` instead.
        is_forecast_for_last_assimilation: bool, optional
            Whether to compute the predictions for the ensemble obtained at the
            last assimilation step. The default is True.
        random_state: Optional[Union[int, np.random.Generator, np.random.RandomState]]
            Pseudorandom number generator state used to generate resamples.
            If `random_state` is ``None`` (or `np.random`), the
            `numpy.random.RandomState` singleton is used.
            If `random_state` is an int, a new ``RandomState`` instance is used,
            seeded with `random_state`.
            If `random_state` is already a ``Generator`` or ``RandomState``
            instance then that instance is used.
        batch_size: int
            Number of parameters that are assimilated at once. This option is
            available to overcome memory limitations when the number of parameters is
            large. In that case, the size of the covariance matrices tends to explode
            and the update step must be performed by chunks of parameters.
            The default is 5000.
        is_parallel_analyse_step: bool, optional
            Whether to use parallel computing for the analyse step if the number of
            batch is above one. It relies on `concurrent.futures` multiprocessing.
            The default is True.
        truncation: float
            A value in the range ]0, 1], used to determine the number of
            significant singular values kept when using svd for the inversion
            of $(C_{dd} + \alpha C_{d})$: Only the largest singular values are kept,
            corresponding to this fraction of the sum of the nonzero singular values.
            The goal of truncation is to deal with smaller matrices (dimensionality
            reduction), easier to inverse. The default is 0.99.
        logger: Optional[logging.Logger]
            Optional :class:`logging.Logger` instance used for event logging.
            The default is None.
        """
        super().__init__(
            obs=obs,
            m_init=m_init,
            cov_obs=cov_obs,
            forward_model=forward_model,
            forward_model_args=forward_model_args,
            forward_model_kwargs=forward_model_kwargs,
            n_assimilations=n_assimilations,
            inversion_type=inversion_type,
            cov_mm_inflation_factor=cov_mm_inflation_factor,
            C_DD_localization=C_DD_localization,
            C_MD_localization=C_MD_localization,
            m_bounds=m_bounds,
            save_ensembles_history=save_ensembles_history,
            seed=seed,
            is_forecast_for_last_assimilation=is_forecast_for_last_assimilation,
            random_state=random_state,
            batch_size=batch_size,
            is_parallel_analyse_step=is_parallel_analyse_step,
            truncation=truncation,
            logger=logger,
        )
        self.set_cov_obs_inflation_factors(cov_obs_inflation_factors)

    @property
    def cov_obs_inflation_factors(self) -> List[float]:
        r"""
        Get the inlfation factors for the covariance matrix of the measurement errors.

        Single and multiple data assimilation are equivalent for the
        linear-Gaussian case as long as the factor :math:`\alpha_{l}` used to
        inflate the covariance matrix of the measurement errors satisfy the
        following condition:

        .. math::
            \sum_{l=1}^{N_{a}} \frac{1}{\alpha_{l}} = 1

        In practise, :math:`\alpha_{l} = N_{a}` is a good choice
        :cite:p:`emerickEnsembleSmootherMultiple2013`.
        """
        return self._cov_obs_inflation_factors

    def set_cov_obs_inflation_factors(self, a: Optional[Sequence[float]]) -> None:
        """Set the inflation factors the covariance matrix of the measurement errors."""
        if a is None:
            self._cov_obs_inflation_factors: List[float] = [
                1 / self.n_assimilations
            ] * self.n_assimilations
        elif len(a) != self.n_assimilations:
            raise ValueError(
                "The length of cov_obs_inflation_factors should match n_assimilations"
            )
        else:
            self._cov_obs_inflation_factors = list(a)

    def solve(self) -> None:
        """Solve the optimization problem with ES-MDA algorithm."""
        if self.save_ensembles_history:
            self.m_history.append(self.m_prior)  # save m_init
        for self._assimilation_step in range(self.n_assimilations):
            self.loginfo(f"Assimilation # {self._assimilation_step + 1}")
            # inflating the covariance
            self._forecast()
            self._pertrub(self.cov_obs_inflation_factors[self._assimilation_step])

            if self.n_batches == 1:
                # Update the prior parameter for next iteration
                self.m_prior = self._apply_bounds(
                    self._analyse(
                        self.cov_obs_inflation_factors[self._assimilation_step]
                    )
                )
            else:
                # Update the prior parameter for next iteration
                self.m_prior = self._apply_bounds(
                    self._local_analyse(
                        self.cov_obs_inflation_factors[self._assimilation_step]
                    )
                )
            # Saving the parameters history
            if self.save_ensembles_history:
                self.m_history.append(self.m_prior)

        if self.is_forecast_for_last_assimilation:
            self._forecast()
