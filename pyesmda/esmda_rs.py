"""
Implement the ES-MDA-RS algorithms.

@author: acollet
"""
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import numpy.typing as npt

from pyesmda.esmda import ESMDA
from pyesmda.utils import compute_ensemble_average_normalized_objective_function

# pylint: disable=C0103 # Does not conform to snake_case naming style


class ESMDA_RS(ESMDA):
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
        (:math:`N_{e}`, :math:`N_{obs}`).
    d_pred: npt.NDArray[np.float64]
        Vectors of predicted values (one for each ensemble member)
        with dimensions (:math:`N_{e}`, :math:`N_{obs}`).
    d_history: List[npt.NDArray[np.float64]]
        List of vectors of predicted values obtained at each assimilation step.
    m_prior:
        Vectors of parameter values (one vector for each ensemble member) used in the
        last assimilation step. Dimensions are (:math:`N_{e}`, :math:`N_{m}`).
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
    cov_obs_inflation_factors : List[float]
        List of multiplication factor used to inflate the covariance matrix of the
        measurement errors. It is determined automatically.
    cov_mm_inflation_factors: List[float]
        List of factors used to inflate the adjusted parameters covariance among
        iterations:
        Each realization of the ensemble at the end of each update step i,
        is linearly inflated around its mean.
        See :cite:p:`andersonExploringNeedLocalization2007`.
    dd_correlation_matrix : Optional[npt.NDArray[np.float64]]
        Correlation matrix based on spatial and temporal distances between
        observations and observations :math:`\rho_{DD}`. It is used to localize the
        autocovariance matrix of predicted data by applying an elementwise
        multiplication by this matrix.
        Expected dimensions are (:math:`N_{obs}`, :math:`N_{obs}`).
    md_correlation_matrix : Optional[npt.NDArray[np.float64]]
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
    """

    # pylint: disable=R0902 # Too many instance attributes
    __slots__: List[str] = ["std_m_prior"]

    def __init__(
        self,
        obs: npt.NDArray[np.float64],
        m_init: npt.NDArray[np.float64],
        cov_obs: npt.NDArray[np.float64],
        std_m_prior: npt.NDArray[np.float64],
        forward_model: Callable[..., npt.NDArray[np.float64]],
        forward_model_args: Sequence[Any] = (),
        forward_model_kwargs: Optional[Dict[str, Any]] = None,
        cov_mm_inflation_factors: Optional[Sequence[float]] = None,
        dd_correlation_matrix: Optional[npt.NDArray[np.float64]] = None,
        md_correlation_matrix: Optional[npt.NDArray[np.float64]] = None,
        m_bounds: Optional[npt.NDArray[np.float64]] = None,
        save_ensembles_history: bool = False,
        seed: Optional[int] = None,
        is_forecast_for_last_assimilation: bool = True,
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
            (:math:`N_{e}`, :math:`N_{m}`).
        cov_obs: npt.NDArray[np.float64]
            Covariance matrix of observed data measurement errors with dimensions
            (:math:`N_{obs}`, :math:`N_{obs}`). Also denoted :math:`R`.
        std_m_prior: npt.NDArray[np.float64]
            Vector of a priori standard deviation :math:`sigma` of the estimated
            parameter. The expected dimension is (:math:`N_{m}`).
            It is the diagonal of :math:`C_{M}`.
        forward_model: callable
            Function calling the non-linear observation model (forward model)
            for all ensemble members and returning the predicted data for
            each ensemble member.
        forward_model_args: Optional[Tuple[Any]]
            Additional args for the callable forward_model. The default is None.
        forward_model_kwargs: Optional[Dict[str, Any]]
            Additional kwargs for the callable forward_model. The default is None.
        cov_mm_inflation_factors: Optional[Sequence[float]]
            List of factors used to inflate the adjusted parameters covariance
            among iterations:
            Each realization of the ensemble at the end of each update step i,
            is linearly inflated around its mean.
            Must match the number of data assimilations (:math:`N_{a}`).
            See :cite:p:`andersonExploringNeedLocalization2007`.
            If None, the default is 1.0. at each iteration (no inflation).
            The default is None.
        dd_correlation_matrix : Optional[npt.NDArray[np.float64]]
            Correlation matrix based on spatial and temporal distances between
            observations and observations :math:`\rho_{DD}`. It is used to localize the
            autocovariance matrix of predicted data by applying an elementwise
            multiplication by this matrix.
            Expected dimensions are (:math:`N_{obs}`, :math:`N_{obs}`).
            The default is None.
        md_correlation_matrix : Optional[npt.NDArray[np.float64]]
            Correlation matrix based on spatial and temporal distances between
            parameters and observations :math:`\rho_{MD}`. It is used to localize the
            cross-covariance matrix between the forecast state vector (parameters)
            and predicted data by applying an elementwise
            multiplication by this matrix.
            Expected dimensions are (:math:`N_{m}`, :math:`N_{obs}`).
            The default is None.
        m_bounds : Optional[npt.NDArray[np.float64]], optional
            Lower and upper bounds for the :math:`N_{m}` parameter values.
            Expected dimensions are (:math:`N_{m}`, 2) with lower bounds on the first
            column and upper on the second one. The default is None.
        save_ensembles_history: bool, optional
            Whether to save the history predictions and parameters over
            the assimilations. The default is False.
        seed: Optional[int]
            Seed for the white noise generator used in the perturbation step.
            If None, the default :func:`numpy.random.default_rng()` is used.
            The default is None.
        is_forecast_for_last_assimilation: bool, optional
            Whether to compute the predictions for the ensemble obtained at the
            last assimilation step. The default is True.

        """
        self.obs: npt.NDArray[np.float64] = obs
        self.m_prior: npt.NDArray[np.float64] = m_init
        self.save_ensembles_history: bool = save_ensembles_history
        self.m_history: list[npt.NDArray[np.float64]] = []
        self.d_history: list[npt.NDArray[np.float64]] = []
        self.d_pred: npt.NDArray[np.float64] = np.zeros([self.n_ensemble, self.d_dim])
        self.cov_obs = cov_obs
        self.std_m_prior: npt.NDArray[np.float64] = std_m_prior
        self.d_obs_uc: npt.NDArray[np.float64] = np.array([])
        self.cov_md: npt.NDArray[np.float64] = np.array([])
        self.cov_dd: npt.NDArray[np.float64] = np.array([])
        self.forward_model: Callable = forward_model
        self.forward_model_args: Sequence[Any] = forward_model_args
        if forward_model_kwargs is None:
            forward_model_kwargs = {}
        self.forward_model_kwargs: Dict[str, Any] = forward_model_kwargs
        self._assimilation_step: int = 0
        self.cov_obs_inflation_factors = []
        self.cov_mm_inflation_factors = cov_mm_inflation_factors
        self.dd_correlation_matrix = dd_correlation_matrix
        self.md_correlation_matrix = md_correlation_matrix
        self.m_bounds = m_bounds
        self.rng: np.random.Generator = np.random.default_rng(seed)
        self.is_forecast_for_last_assimilation = is_forecast_for_last_assimilation

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
        while not self._is_unity_reached(current_inflation_factor):
            self._assimilation_step += 1
            print(f"Assimilation # {self._assimilation_step}")
            self._forecast()
            # Divide per 2, because it is multiplied by 2 as the beginning
            # of the second while loop
            current_inflation_factor: float = (
                self._compute_initial_inflation_factor() / 2
            )
            is_valid_parameter_change: bool = False
            while not is_valid_parameter_change:
                current_inflation_factor *= 2  # double the inflation (dumping) factor
                self._pertrub(current_inflation_factor)
                self._approximate_covariance_matrices()
                m_pred = self._apply_bounds(self._analyse(current_inflation_factor))
                is_valid_parameter_change: bool = self._is_valid_parameter_change(
                    m_pred
                )

            # If the criteria is reached -> Get exactly one for the sum
            if self._is_unity_reached(current_inflation_factor):
                current_inflation_factor = 1 / (
                    1 - np.sum([1 / a for a in self.cov_obs_inflation_factors])
                )
                self._pertrub(current_inflation_factor)
                self._approximate_covariance_matrices()
                m_pred = self._analyse(current_inflation_factor)
                is_valid_parameter_change: bool = self._is_valid_parameter_change(
                    m_pred
                )

            self.cov_obs_inflation_factors.append(current_inflation_factor)
            print(f"- Inflation factor = {current_inflation_factor:.3f}")

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
        return 0.25 * compute_ensemble_average_normalized_objective_function(
            self.d_pred, self.obs, self.cov_obs
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
        return (
            np.sum([1 / a for a in self.cov_obs_inflation_factors])
            + 1 / current_inflation_factor
            >= 1
        )

    def _is_valid_parameter_change(self, m_pred: npt.NDArray[np.float64]) -> bool:
        """Check if all change residuals are below 2 sigma.

        Parameters
        ----------
        m_pred : npt.NDArray[np.float64]
            _description_

        Returns
        -------
        bool
            _description_
        """

        def is_lower(residuals) -> bool:
            return np.all(residuals < 2 * self.std_m_prior)

        return np.all(list(map(is_lower, np.abs(m_pred - self.m_prior))))
