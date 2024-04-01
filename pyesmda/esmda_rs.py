"""
Implement the ES-MDA-RS algorithms.

@author: acollet
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt
from scipy.sparse import spmatrix  # type: ignore

from pyesmda.esmda import ESMDABase
from pyesmda.inversion import ESMDAInversionType
from pyesmda.utils import NDArrayFloat, ls_cost_function

# pylint: disable=C0103 # Does not conform to snake_case naming style


class ESMDA_RS(ESMDABase):
    r"""
    Restricted Step Ensemble Smoother with Multiple Data Assimilation.

    Implement an adaptative version of the original ES-MDA algorithm proposed by
    Emerick, A. A. and A. C. Reynolds
    :cite:p:`emerickEnsembleSmootherMultiple2013,
    emerickHistoryMatchingProductionSeismic2013`. This adaptative version introduced by
    :cite:p:`leAdaptiveEnsembleSmoother2016` provides an automatic procedure for
    choosing the inflation factor for the next data-assimilation step adaptively
    as the history match proceeds. The procedure also decides when to stop,
    i.e. the number of assimilation, which is no longer a user input.

    Attributes
    ----------
    d_dim : int
        Number of observation values :math:`N_{obs}`, and consequently of
        predicted values.
    obs : npt.NDArray[np.float64]
        Obsevrations vector with dimensions (:math:`N_{obs}`).
    cov_obs: npt.NDArray[np.float64]
        Covariance matrix of observed data measurement errors with dimensions
        (:math:`N_{obs}`, :math:`N_{obs}`). Also denoted :math:`R`.
    std_m_prior: npt.NDArray[np.float64]
        Vector of a priori standard deviation :math:`sigma` of the estimated
        parameter. The expected dimension is (:math:`N_{m}`).
        It is the diagonal of :math:`C_{M}`.
    d_obs_uc: npt.NDArray[np.float64]
        Vectors of pertubed observations with dimension
        (:math:`N_{obs}`, :math:`N_{e}`).
    d_pred: npt.NDArray[np.float64]
        Vectors of predicted values (one for each ensemble member)
        with dimensions (:math:`N_{obs}`, :math:`N_{e}`).
    d_history: List[npt.NDArray[np.float64]]
        List of vectors of predicted values obtained at each assimilation step.
    m_prior:
        Vectors of parameter values (one vector for each ensemble member) used in the
        last assimilation step. Dimensions are (:math:`N_{m}`, :math:`N_{e}`).
    m_bounds : npt.NDArray[np.float64]
        Lower and upper bounds for the :math:`N_{m}` parameter values.
        Expected dimensions are (:math:`N_{m}`, 2) with lower bounds on the first
        column and upper on the second one.
    m_history: List[npt.NDArray[np.float64]]
        List of successive `m_prior`.
    cov_md: npt.NDArray[np.float64]
        Cross-covariance matrix between the forecast state vector and predicted data.
        Dimensions are (:math:`N_{m}, N_{obs}`).
    cov_dd: npt.NDArray[np.float64]
        Autocovariance matrix of predicted data.
        Dimensions are (:math:`N_{obs}, N_{obs}`).
    cov_mm: npt.NDArray[np.float64]
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
        Number of data assimilations (:math:`N_{a}`) performed.
        Automatically determined. Initially at 0.
    cov_mm_inflation_factor: float
        Factor used to inflate the initial ensemble around its mean.
        See :cite:p:`andersonExploringNeedLocalization2007`.
    dd_correlation_matrix : Optional[csr_matrix]
        Correlation matrix based on spatial and temporal distances between
        observations and observations :math:`\rho_{DD}`. It is used to localize the
        autocovariance matrix of predicted data by applying an elementwise
        multiplication by this matrix.
        Expected dimensions are (:math:`N_{obs}`, :math:`N_{obs}`).
    md_correlation_matrix : Optional[spmatrix]
        Correlation matrix based on spatial and temporal distances between
        parameters and observations :math:`\rho_{MD}`. It is used to localize the
        cross-covariance matrix between the forecast state vector (parameters)
        and predicted data by applying an elementwise
        multiplication by this matrix.
        Expected dimensions are (:math:`N_{m}`, :math:`N_{obs}`).
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
    __slots__: List[str] = ["std_m_prior", "_cov_obs_inflation_factors"]

    def __init__(
        self,
        obs: npt.NDArray[np.float64],
        m_init: npt.NDArray[np.float64],
        cov_obs: npt.NDArray[np.float64],
        forward_model: Callable[..., npt.NDArray[np.float64]],
        forward_model_args: Sequence[Any] = (),
        forward_model_kwargs: Optional[Dict[str, Any]] = None,
        std_m_prior: Optional[npt.NDArray[np.float64]] = None,
        inversion_type: Union[ESMDAInversionType, str] = ESMDAInversionType.NAIVE,
        cov_mm_inflation_factor: float = 1.0,
        dd_correlation_matrix: Optional[Union[NDArrayFloat, spmatrix]] = None,
        md_correlation_matrix: Optional[Union[NDArrayFloat, spmatrix]] = None,
        m_bounds: Optional[npt.NDArray[np.float64]] = None,
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
        obs : npt.NDArray[np.float64]
            Obsevrations vector with dimension :math:`N_{obs}`.
        m_init : npt.NDArray[np.float64]
            Initial ensemble of parameters vector with dimensions
            (:math:`N_{m}`, :math:`N_{e}`).
        cov_obs: npt.NDArray[np.float64]
            Covariance matrix of observed data measurement errors with dimensions
            (:math:`N_{obs}`, :math:`N_{obs}`). Also denoted :math:`R`.
        forward_model: callable
            Function calling the non-linear observation model (forward model)
            for all ensemble members and returning the predicted data for
            each ensemble member.
        forward_model_args: Optional[Tuple[Any]]
            Additional args for the callable forward_model. The default is None.
        forward_model_kwargs: Optional[Dict[str, Any]]
            Additional kwargs for the callable forward_model. The default is None.
        std_m_prior: Optional[npt.NDArray[np.float64]]
            Vector of a priori standard deviation :math:`sigma` of the estimated
            parameter. The expected dimension is (:math:`N_{m}`).
            It is the diagonal of :math:`C_{M}`. If not provided, then it is inffered
            from the inflated initial ensemble (see `cov_mm_inflation_factor`).
            The default is None.
        cov_mm_inflation_factor: float
            Factor used to inflate the initial ensemble variance around its mean.
            See :cite:p:`andersonExploringNeedLocalization2007`.
            The default is 1.0, which means no inflation.
        dd_correlation_matrix : Optional[Union[NDArrayFloat, spmatrix]]
            Correlation matrix based on spatial and temporal distances between
            observations and observations :math:`\rho_{DD}`. It is used to localize the
            autocovariance matrix of predicted data by applying an elementwise
            multiplication by this matrix.
            Expected dimensions are (:math:`N_{obs}`, :math:`N_{obs}`).
            The default is None.
        md_correlation_matrix : Optional[Union[NDArrayFloat, spmatrix]]
            Correlation matrix based on spatial and temporal distances between
            parameters and observations :math:`\rho_{MD}`. It is used to localize the
            cross-covariance matrix between the forecast state vector (parameters)
            and predicted data by applying an elementwise
            multiplication by this matrix.
            Expected dimensions are (:math:`N_{m}`, :math:`N_{obs}`).
            The default is None.
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
            n_assimilations=1,  # in esmda-rs this number is determined automatically
            inversion_type=inversion_type,
            cov_mm_inflation_factor=cov_mm_inflation_factor,
            dd_correlation_matrix=dd_correlation_matrix,
            md_correlation_matrix=md_correlation_matrix,
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

        # Initialize an empty list
        self.cov_obs_inflation_factors = []

        # I am still wondering whether this should remain constant of if it should be
        # updated at each iteration ? I still have a doubt. I asked the authors of the
        # paper and I am still waiting for the answer
        if std_m_prior is not None:
            # in that case the user impose the ensemble variance
            self.std_m_prior: npt.NDArray[np.float64] = std_m_prior
        else:
            # otherwise, it is inffered from the inflated ensemble
            self.std_m_prior = np.std(self.m_prior, axis=1, ddof=1)

    @property
    def n_assimilations(self) -> int:
        """Get the number of assimilations performed. Read-only."""
        return self._assimilation_step

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

    @cov_obs_inflation_factors.setter
    def cov_obs_inflation_factors(self, a: List[float]) -> None:
        """Set the inflation factors the covariance matrix of the measurement errors."""
        self._cov_obs_inflation_factors = a

    def solve(self) -> None:
        """Solve the optimization problem with ES-MDA-RS algorithm."""
        if self.save_ensembles_history:
            self.m_history.append(self.m_prior)  # save m_init

        current_inflation_factor: float = 10.0  # to initiate the while
        m_pred = self.m_prior
        while not self._is_unity_reached(current_inflation_factor):
            self._assimilation_step += 1
            self.loginfo(f"Assimilation # {self._assimilation_step}")
            self._forecast()
            # Divide per 2, because it is multiplied by 2 as the beginning
            # of the second while loop
            current_inflation_factor = self._compute_initial_inflation_factor() / 2
            is_valid_parameter_change: bool = False
            while not is_valid_parameter_change:
                current_inflation_factor *= 2  # double the inflation (dumping) factor
                self._pertrub(current_inflation_factor)

                if self.n_batches == 1:
                    m_pred = self._apply_bounds(self._analyse(current_inflation_factor))
                else:
                    # Update the prior parameter for next iteration
                    m_pred = self._apply_bounds(
                        self._local_analyse(current_inflation_factor)
                    )

                is_valid_parameter_change = self._is_valid_parameter_change(m_pred)

            # If the criteria is reached -> Get exactly one for the sum
            if self._is_unity_reached(current_inflation_factor):
                current_inflation_factor = 1 / (
                    1 - float(np.sum([1 / a for a in self.cov_obs_inflation_factors]))
                )
                self._pertrub(current_inflation_factor)
                m_pred = self._analyse(current_inflation_factor)
                is_valid_parameter_change = self._is_valid_parameter_change(m_pred)

            self.cov_obs_inflation_factors.append(current_inflation_factor)
            self.loginfo(f"- Inflation factor = {current_inflation_factor:.3f}")

            # Update the prior parameter for next iteration
            self.m_prior = m_pred
            # Saving the parameters history
            if self.save_ensembles_history:
                self.m_history.append(m_pred)

        # Last assimilation
        if self.is_forecast_for_last_assimilation:
            self._forecast()

    def _compute_initial_inflation_factor(self) -> float:
        r"""Compute the :math:`\alpha_{l}` inflation (dumping) factor."""
        return (
            0.25
            / self.obs.size
            * float(
                np.mean(ls_cost_function(self.d_pred, self.obs, self.cov_obs_cholesky))
            )
        )

    def _is_unity_reached(self, current_inflation_factor: float) -> bool:
        """
        Whether the sum of the inverse inflation factors is above one.

        It includes all factors up to the current iteration.

        Parameters
        ----------
        current_inflation_factor: float
            Multiplication factor used to inflate the covariance matrix of the
            measurement errors for the current (last) iteration.
        """
        return bool(
            np.sum([1 / a for a in self.cov_obs_inflation_factors])
            + 1 / current_inflation_factor
            >= 1
        )

    def _is_valid_parameter_change(self, m_pred: npt.NDArray[np.float64]) -> bool:
        r"""Check if all change residuals are below 2 sigma.

        Parameters
        ----------
        m_pred : npt.NDArray[np.float64]
            Ensemble of predicted values with dimensions
            (:math:`N_{s}`, :math:`N_{e}`).

        Returns
        -------
        bool
            Whether the parameter change is valid.
        """

        def is_lower(residuals: NDArrayFloat) -> bool:
            return bool(np.all(residuals < 2 * self.std_m_prior))

        return bool(
            np.all(list(map(is_lower, np.abs(m_pred - self.m_prior).T)))  # type: ignore
        )
