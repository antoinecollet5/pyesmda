"""
Implement the ES-MDA algorithms.

@author: acollet
"""
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Union

import numpy as np
from scipy.sparse import csr_matrix

from pyesmda.utils import (
    NDArrayFloat,
    approximate_cov_mm,
    approximate_covariance_matrix_from_ensembles,
    check_nans_in_predictions,
    inflate_ensemble_around_its_mean,
)

# pylint: disable=C0103 # Does not conform to snake_case naming style


class ESMDA:
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
        (:math:`N_{e}`, :math:`N_{obs}`).
    d_pred: NDArrayFloat
        Vectors of predicted values (one for each ensemble member)
        with dimensions (:math:`N_{e}`, :math:`N_{obs}`).
    d_history: List[NDArrayFloat]
        List of vectors of predicted values obtained at each assimilation step.
    m_prior:
        Vectors of parameter values (one vector for each ensemble member) used in the
        last assimilation step. Dimensions are (:math:`N_{e}`, :math:`N_{m}`).
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
    cov_mm_inflation_factors: List[float]
        List of factors used to inflate the adjusted parameters covariance among
        iterations:
        Each realization of the ensemble at the end of each update step i,
        is linearly inflated around its mean.
        See :cite:p:`andersonExploringNeedLocalization2007`.
    dd_correlation_matrix : Optional[Union[NDArrayFloat, csr_matrix]]
        Correlation matrix based on spatial and temporal distances between
        observations and observations :math:`\rho_{DD}`. It is used to localize the
        autocovariance matrix of predicted data by applying an elementwise
        multiplication by this matrix.
        Expected dimensions are (:math:`N_{obs}`, :math:`N_{obs}`). A sparse matrix
        format can be provided to save some memory.
    md_correlation_matrix : Optional[Union[NDArrayFloat, csr_matrix]]
        Correlation matrix based on spatial and temporal distances between
        parameters and observations :math:`\rho_{MD}`. It is used to localize the
        cross-covariance matrix between the forecast state vector (parameters)
        and predicted data by applying an elementwise
        multiplication by this matrix.
        Expected dimensions are (:math:`N_{m}`, :math:`N_{obs}`). . A sparse matrix
        format can be provided to save some memory.
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

    """
    # pylint: disable=R0902 # Too many instance attributes
    __slots__: List[str] = [
        "obs",
        "_cov_obs",
        "d_obs_uc",
        "d_pred",
        "d_history",
        "m_prior",
        "_m_bounds",
        "m_history",
        "cov_md",
        "cov_dd",
        "forward_model",
        "forward_model_args",
        "forward_model_kwargs",
        "_n_assimilations",
        "_assimilation_step",
        "_cov_obs_inflation_factors",
        "_cov_mm_inflation_factors",
        "_dd_correlation_matrix",
        "_md_correlation_matrix",
        "save_ensembles_history",
        "rng",
        "is_forecast_for_last_assimilation",
        "batch_size",
        "is_parallel_analyse_step",
    ]

    def __init__(
        self,
        obs: NDArrayFloat,
        m_init: NDArrayFloat,
        cov_obs: Union[NDArrayFloat, csr_matrix],
        forward_model: Callable[..., NDArrayFloat],
        forward_model_args: Sequence[Any] = (),
        forward_model_kwargs: Optional[Dict[str, Any]] = None,
        n_assimilations: int = 4,
        cov_obs_inflation_factors: Optional[Sequence[float]] = None,
        cov_mm_inflation_factors: Optional[Sequence[float]] = None,
        dd_correlation_matrix: Optional[Union[NDArrayFloat, csr_matrix]] = None,
        md_correlation_matrix: Optional[Union[NDArrayFloat, csr_matrix]] = None,
        m_bounds: Optional[NDArrayFloat] = None,
        save_ensembles_history: bool = False,
        seed: Optional[int] = None,
        is_forecast_for_last_assimilation: bool = True,
        batch_size: int = 5000,
        is_parallel_analyse_step: bool = True,
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
            (:math:`N_{e}`, :math:`N_{m}`).
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
        cov_mm_inflation_factors: Optional[Sequence[float]]
            List of factors used to inflate the adjusted parameters covariance
            among iterations:
            Each realization of the ensemble at the end of each update step i,
            is linearly inflated around its mean.
            Must match the number of data assimilations (:math:`N_{a}`).
            See :cite:p:`andersonExploringNeedLocalization2007`.
            If None, the default is 1.0. at each iteration (no inflation).
            The default is None.
        dd_correlation_matrix : Optional[NDArrayFloat]
            Correlation matrix based on spatial and temporal distances between
            observations and observations :math:`\rho_{DD}`. It is used to localize the
            autocovariance matrix of predicted data by applying an elementwise
            multiplication by this matrix.
            Expected dimensions are (:math:`N_{obs}`, :math:`N_{obs}`).
            The default is None.
        md_correlation_matrix : Optional[NDArrayFloat]
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
            Seed for the white noise generator used in the perturbation step.
            If None, the default :func:`numpy.random.default_rng()` is used.
            The default is None.
        is_forecast_for_last_assimilation: bool, optional
            Whether to compute the predictions for the ensemble obtained at the
            last assimilation step. The default is True.
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

        """
        self.obs: NDArrayFloat = obs
        self.m_prior: NDArrayFloat = m_init
        self.save_ensembles_history: bool = save_ensembles_history
        self.m_history: list[NDArrayFloat] = []
        self.d_history: list[NDArrayFloat] = []
        self.d_pred: NDArrayFloat = np.zeros([self.n_ensemble, self.d_dim])
        self.cov_obs = cov_obs
        self.d_obs_uc: NDArrayFloat = np.array([])
        self.cov_md: NDArrayFloat = np.array([])
        self.cov_dd: NDArrayFloat = np.array([])
        self.forward_model: Callable = forward_model
        self.forward_model_args: Sequence[Any] = forward_model_args
        if forward_model_kwargs is None:
            forward_model_kwargs = {}
        self.forward_model_kwargs: Dict[str, Any] = forward_model_kwargs
        self._set_n_assimilations(n_assimilations)
        self._assimilation_step: int = 0
        self.set_cov_obs_inflation_factors(cov_obs_inflation_factors)
        self.cov_mm_inflation_factors = cov_mm_inflation_factors
        self.dd_correlation_matrix = dd_correlation_matrix
        self.md_correlation_matrix = md_correlation_matrix
        self.m_bounds = m_bounds
        self.rng: np.random.Generator = np.random.default_rng(seed)
        self.is_forecast_for_last_assimilation = is_forecast_for_last_assimilation
        self.batch_size = batch_size
        self.is_parallel_analyse_step = is_parallel_analyse_step

    @property
    def n_assimilations(self) -> int:
        """Return the number of assimilations to perform. Read-only attribute."""
        return self._n_assimilations

    def _set_n_assimilations(self, n: int) -> None:
        """Set the number of assimilations to perform."""
        try:
            if int(n) < 1:
                raise ValueError("The number of assimilations must be 1 or more.")
            if int(n) != float(n):
                raise TypeError()
        except TypeError as e:
            raise TypeError(
                "The number of assimilations must be a positive integer."
            ) from e

        self._n_assimilations = int(n)

    @property
    def n_ensemble(self) -> int:
        """Return the number of ensemble members."""
        return self.m_prior.shape[0]

    @property
    def m_dim(self) -> int:
        """Return the length of the parameters vector."""
        return self.m_prior.shape[1]

    @property
    def d_dim(self) -> int:
        """Return the number of forecast data."""
        return len(self.obs)

    @property
    def cov_obs(self) -> csr_matrix:
        """Get the observation errors covariance matrix."""
        return self._cov_obs

    @cov_obs.setter
    def cov_obs(self, s: Union[NDArrayFloat, csr_matrix]) -> None:
        """Set the observation errors covariance matrix."""
        if len(s.shape) != 2 or s.shape[0] != s.shape[1] or s.shape[0] != self.d_dim:
            raise ValueError(
                "cov_obs must be a 2D square matrix with "
                f"dimensions ({self.d_dim}, {self.d_dim})."
            )
        self._cov_obs: csr_matrix = csr_matrix(s)

    @property
    def cov_mm(self) -> NDArrayFloat:
        r"""
        Get the estimated parameters autocovariance matrix. It is a read-only attribute.

        The covariance matrice :math:`C^{l}_{MM}`
        is approximated from the ensemble in the standard way of EnKF
        :cite:p:`evensenDataAssimilationEnsemble2007,aanonsenEnsembleKalmanFilter2009`:

        .. math::
           C^{l}_{MM} = \frac{1}{N_{e} - 1} \sum_{j=1}^{N_{e}}\left(m^{l}_{j} -
           \overline{m^{l}}\right)\left(m^{l}_{j}
           - \overline{m^{l}} \right)^{T}

        with :math:`\overline{m^{l}}`, the parameters
        ensemble means, at iteration :math:`l`.
        """
        return approximate_cov_mm(self.m_prior)

    @property
    def m_bounds(self) -> NDArrayFloat:
        """Get the parameter errors covariance matrix."""
        return self._m_bounds

    @m_bounds.setter
    def m_bounds(self, mb: Optional[NDArrayFloat]) -> None:
        """Set the parameter errors covariance matrix."""
        if mb is None:
            # In that case, create an array of nan.
            self._m_bounds: NDArrayFloat = np.empty([self.m_dim, 2], dtype=np.float64)
            self._m_bounds[:] = np.nan
        elif mb.shape[0] != self.m_dim:
            raise ValueError(
                f"m_bounds is of shape {mb.shape} while it "
                f"should be of shape ({self.m_dim}, 2)"
            )
        else:
            self._m_bounds = mb

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

    @property
    def cov_mm_inflation_factors(self) -> List[float]:
        r"""
        Get the inlfation factors for the adjusted parameters covariance matrix.

        Covariance inflation is a method used to counteract the tendency of ensemble
        Kalman methods to underestimate the uncertainty because of either undersampling,
        inbreeding, or spurious correlations
        :cite:p:`todaroAdvancedTechniquesSolving2021`.
        The spread of the ensemble is artificially increased before the assimilation
        of the observations, according to scheme introduced by
        :cite:`andersonExploringNeedLocalization2007`:

        .. math::
            m^{l}_{j} \leftarrow r^{l}\left(m^{l}_{j} - \frac{1}{N_{e}}
            \sum_{j}^{N_{e}}m^{l}_{j}\right) + \frac{1}{N_{e}}\sum_{j}^{N_{e}}m^{l}_{j}

        where :math:`r` is the inflation factor for the assimilation step :math:`l`.
        """
        return list(self._cov_mm_inflation_factors)

    @cov_mm_inflation_factors.setter
    def cov_mm_inflation_factors(self, a: Optional[Sequence[float]]) -> None:
        """
        Set the inflation factors the adjusted parameters covariance matrix.

        If no values have been provided by the user, the default is 1.0. at
        each iteration (no inflation).
        """
        if a is None:
            self._cov_mm_inflation_factors: List[float] = [1.0] * self.n_assimilations
        elif len(a) != self.n_assimilations:
            raise ValueError(
                "The length of cov_mm_inflation_factors should match n_assimilations"
            )
        else:
            self._cov_mm_inflation_factors = list(a)

    @property
    def dd_correlation_matrix(self) -> Optional[csr_matrix]:
        """Get the observations-observations localization matrix."""
        return self._dd_correlation_matrix

    @dd_correlation_matrix.setter
    def dd_correlation_matrix(
        self, s: Optional[Union[NDArrayFloat, csr_matrix]]
    ) -> None:
        """Set the observations-observations localization matrix."""
        if s is None:
            self._dd_correlation_matrix = None
            return
        if len(s.shape) != 2 or s.shape[0] != s.shape[1] or s.shape[0] != self.d_dim:
            raise ValueError(
                "dd_correlation_matrix must be a 2D square matrix with "
                f"dimensions ({self.d_dim}, {self.d_dim})."
            )
        self._dd_correlation_matrix: Optional[csr_matrix] = csr_matrix(s)

    @property
    def md_correlation_matrix(self) -> Optional[csr_matrix]:
        """Get the parameters-observations localization matrix."""
        return self._md_correlation_matrix

    @md_correlation_matrix.setter
    def md_correlation_matrix(
        self, s: Optional[Union[NDArrayFloat, csr_matrix]]
    ) -> None:
        """Set the parameters-observations localization matrix."""
        if s is None:
            self._md_correlation_matrix = None
            return
        if len(s.shape) != 2:
            raise ValueError(
                "md_correlation_matrix must be a 2D matrix with "
                f"dimensions ({self.m_dim}, {self.d_dim})."
            )
        if s.shape[0] != self.m_dim or s.shape[1] != self.d_dim:
            raise ValueError(
                "md_correlation_matrix must be a 2D matrix with "
                f"dimensions ({self.m_dim}, {self.d_dim})."
            )
        self._md_correlation_matrix: Optional[csr_matrix] = csr_matrix(s)

    @property
    def n_batches(self) -> int:
        """Number of batch used in the optimization."""
        return int(self.m_dim / self.batch_size) + 1

    def solve(self) -> None:
        """Solve the optimization problem with ES-MDA algorithm."""
        if self.n_batches == 1:
            self._solve()
        else:
            # assimilate chunk of parameters rather than everything all at once.
            self._solve_locally()

    def _solve(self) -> None:
        """Solve the optimization problem with ES-MDA algorithm."""
        if self.save_ensembles_history:
            self.m_history.append(self.m_prior)  # save m_init
        for self._assimilation_step in range(self.n_assimilations):
            print(f"Assimilation # {self._assimilation_step + 1}")
            # inflating the covariance
            self.m_prior = self._apply_bounds(
                inflate_ensemble_around_its_mean(
                    self.m_prior,
                    self.cov_mm_inflation_factors[self._assimilation_step],
                )
            )
            self._forecast()
            self._pertrub(self.cov_obs_inflation_factors[self._assimilation_step])
            self._approximate_covariance_matrices()
            # Update the prior parameter for next iteration
            self.m_prior = self._apply_bounds(
                self._analyse(self.cov_obs_inflation_factors[self._assimilation_step])
            )
            # Saving the parameters history
            if self.save_ensembles_history:
                self.m_history.append(self.m_prior)

        if self.is_forecast_for_last_assimilation:
            self._forecast()

    def _solve_locally(self) -> None:
        """Solve the optimization problem with ES-MDA algorithm."""
        if self.save_ensembles_history:
            self.m_history.append(self.m_prior)  # save m_init
        for self._assimilation_step in range(self.n_assimilations):
            print(f"Assimilation # {self._assimilation_step + 1}")
            self._forecast()
            self._pertrub(self.cov_obs_inflation_factors[self._assimilation_step])

            # covariance approximation dd
            self.cov_dd = approximate_covariance_matrix_from_ensembles(
                self.d_pred, self.d_pred
            )
            # Spatial and temporal localization: obs - obs
            if self.dd_correlation_matrix is not None:
                self.cov_dd = self.dd_correlation_matrix.multiply(self.cov_dd).toarray()

            # Update the prior parameter for next iteration
            self.m_prior = self._apply_bounds(
                inflate_ensemble_around_its_mean(
                    self._local_analyse(
                        self.cov_obs_inflation_factors[self._assimilation_step]
                    ),
                    self.cov_mm_inflation_factors[self._assimilation_step],
                )
            )

            # Saving the parameters history
            if self.save_ensembles_history:
                self.m_history.append(self.m_prior)

        if self.is_forecast_for_last_assimilation:
            self._forecast()

    def _forecast(self) -> None:
        r"""
        Forecast step of ES-MDA.

        Run the forward model from time zero until the end of the historical
        period from time zero until the end of the historical period to
        compute the vector of predicted data

        .. math::
            d^{l}_{j}=g\left(m^{l}_{j}\right),\textrm{for }j=1,2,...,N_{e},

        where :math:`g(·)` denotes the nonlinear observation model, i.e.,
        :math:`d^{l}_{j}` is the :math:`N_{d}`-dimensional vector of predicted
        data obtained by running
        the forward model reservoir simulation with the model parameters given
        by the vector :math:`m^{l}_{j}` from time zero. Note that we use
        :math:`N_{d}` to denote the total number of measurements in the entire
        history.
        """
        self.d_pred = self.forward_model(
            self.m_prior, *self.forward_model_args, **self.forward_model_kwargs
        )
        if self.save_ensembles_history:
            self.d_history.append(self.d_pred)

        # Check if no nan values are found in the predictions.
        # If so, stop the assimilation
        check_nans_in_predictions(self.d_pred, self._assimilation_step)

    def _pertrub(self, inflation_factor: float) -> None:
        r"""
        Perturbation of the observation vector step of ES-MDA.

        Perturb the vector of observations

        .. math::
            d^{l}_{uc,j} = d_{obs} + \sqrt{\alpha_{l+1}}C_{D}^{1/2}Z_{d},
            \textrm{for } j=1,2,...,N_{e},

        where :math:`Z_{d} \sim \mathcal{N}(O, I_{N_{d}})`.

        Notes
        -----
        To get reproducible behavior, use a seed when creating the ESMDA instance.

        """
        self.d_obs_uc = np.zeros([self.n_ensemble, self.d_dim])
        for i in range(self.d_dim):
            self.d_obs_uc[:, i] = self.obs[i] + np.sqrt(
                inflation_factor
            ) * self.rng.normal(0, np.abs(self.cov_obs.diagonal()[i]), self.n_ensemble)

    def _approximate_covariance_matrices(self) -> None:
        r"""
        Approximate the covariance matrices.

        The covariance matrices :math:`C^{l}_{MD}` and :math:`C^{l}_{DD}`
        are approximated from the ensemble in the standard way of EnKF
        :cite:p:`evensenDataAssimilationEnsemble2007,aanonsenEnsembleKalmanFilter2009`:

        .. math::
           C^{l}_{MD} = \frac{1}{N_{e} - 1} \sum_{j=1}^{N_{e}}\left(m^{l}_{j} -
           \overline{m^{l}}\right)\left(d^{l}_{j}
           - \overline{d^{l}} \right)^{T}

        .. math::
           C^{l}_{DD} = \frac{1}{N_{e} - 1} \sum_{j=1}^{N_{e}}\left(d^{l}_{j}
           -\overline{d^{l}} \right)\left(d^{l}_{j}
           - \overline{d^{l}} \right)^{T}

        with :math:`\overline{m^{l}}` and :math:`\overline{d^{l}}`, the
        the ensemble means, at iteration :math:`l`, of parameters and predictions,
        respectively.

        If defined by the user, covariances localization is applied by element-wise
        multiplication (Schur product or Hadamard product) of the original
        covariance matrix and a distance dependent correlation function
        that smoothly reduces the correlations between points for increasing distances
        and cuts off long-range correlations above a specific distance:

        .. math::
           \tilde{C}^{l}_{MD} = \rho_{MD} \odot C^{l}_{MD}

        .. math::
           \tilde{C}^{l}_{DD} = \rho_{DD} \odot C^{l}_{DD}

        with :math:`\odot` the element wise multiplication.
        """
        self.cov_md = approximate_covariance_matrix_from_ensembles(
            self.m_prior, self.d_pred
        )
        self.cov_dd = approximate_covariance_matrix_from_ensembles(
            self.d_pred, self.d_pred
        )

        # Spatial and temporal localization: obs - obs
        if self.dd_correlation_matrix is not None:
            self.cov_dd = self.dd_correlation_matrix.multiply(self.cov_dd).toarray()
        # Spatial and temporal localization: parameters - obs
        if self.md_correlation_matrix is not None:
            self.cov_md = self.md_correlation_matrix.multiply(self.cov_md).toarray()

    def _analyse(self, inflation_factor: float) -> NDArrayFloat:
        r"""
        Analysis step of the ES-MDA.

        Update the vector of model parameters using

        .. math::
           m^{l+1}_{j} = m^{l}_{j} + C^{l}_{MD}\left(C^{l}_{DD}+\alpha_{l+1}
           C_{D}\right)^{-1} \left(d^{l}_{uc,j} - d^{l}_{j} \right),
           \textrm{for } j=1,2,...,N_{e}.

        Notes
        -----
        To avoid the inversion of :math:`\left(C^{l}_{DD}+\alpha_{l+1} C_{D}\right)`,
        the product :math:`\left(C^{l}_{DD}+\alpha_{l+1} C_{D}\right) ^{-1}
        \left(d^{l}_{uc,j} - d^{l}_{j} \right)`
        is solved linearly as :math:`A^{-1}b = x`
        which is equivalent to solve :math:`Ax = b`.

        """
        # predicted parameters
        return (
            self.m_prior
            + (
                self.cov_md
                @ np.linalg.solve(
                    self.cov_dd + inflation_factor * self.cov_obs,
                    (self.d_obs_uc - self.d_pred).T,
                )
            ).T
        )

    def _local_analyse(self, inflation_factor: float) -> NDArrayFloat:
        r"""
        Analysis step of the ES-MDA.

        Update the vector of model parameters using

        .. math::
           m^{l+1}_{j} = m^{l}_{j} + C^{l}_{MD}\left(C^{l}_{DD}+\alpha_{l+1}
           C_{D}\right)^{-1} \left(d^{l}_{uc,j} - d^{l}_{j} \right),
           \textrm{for } j=1,2,...,N_{e}.

        Notes
        -----
        To avoid the inversion of :math:`\left(C^{l}_{DD}+\alpha_{l+1} C_{D}\right)`,
        the product :math:`\left(C^{l}_{DD}+\alpha_{l+1} C_{D}\right) ^{-1}
        \left(d^{l}_{uc,j} - d^{l}_{j} \right)`
        is solved linearly as :math:`A^{-1}b = x`
        which is equivalent to solve :math:`Ax = b`.

        """
        m_pred: NDArrayFloat = np.zeros(self.m_prior.shape)

        if self.is_parallel_analyse_step:
            with ProcessPoolExecutor() as executor:
                results: Iterator[NDArrayFloat] = executor.map(
                    self._get_batch_m_update,
                    range(self.n_batches),
                    [inflation_factor] * self.n_batches,
                )
                for index, res in enumerate(results):
                    _slice = slice(
                        index * self.batch_size, (index + 1) * self.batch_size
                    )
                    m_pred[:, _slice] = res
            # self.simu_n += n_ensemble
        else:
            for index in range(self.n_batches):
                _slice = slice(index * self.batch_size, (index + 1) * self.batch_size)
                m_pred[:, _slice] = self._get_batch_m_update(index, inflation_factor)

        return m_pred

    def _get_batch_m_update(self, index: int, inflation_factor: float) -> NDArrayFloat:
        _slice = slice(index * self.batch_size, (index + 1) * self.batch_size)
        batch_cov_md = approximate_covariance_matrix_from_ensembles(
            self.m_prior[:, _slice], self.d_pred
        )
        # apply localization
        if self.md_correlation_matrix is not None:
            batch_cov_md = self.md_correlation_matrix[_slice, :].multiply(batch_cov_md)
        return (
            self.m_prior[:, _slice]
            + (
                batch_cov_md
                @ np.linalg.solve(
                    self.cov_dd + inflation_factor * self.cov_obs,
                    (self.d_obs_uc - self.d_pred).T,
                )
            ).T
        )

    def _apply_bounds(self, m_pred: NDArrayFloat) -> NDArrayFloat:
        """Apply bounds constraints to the adjusted parameters."""
        m_pred = np.where(
            m_pred < self.m_bounds[:, 0], self.m_bounds[:, 0], m_pred
        )  # lower bounds
        m_pred = np.where(
            m_pred > self.m_bounds[:, 1], self.m_bounds[:, 1], m_pred
        )  # upper bounds

        return m_pred
