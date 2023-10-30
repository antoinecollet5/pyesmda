"""
Implement a base class for the ES-MDA algorithms and variants.

@author: acollet
"""
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Union

import numpy as np
import scipy as sp  # type: ignore
from scipy._lib._util import check_random_state  # type: ignore
from scipy.sparse import csr_matrix, spmatrix  # type: ignore

from pyesmda.inversion import ESMDAInversionType, inversion
from pyesmda.utils import NDArrayFloat, check_nans_in_predictions, get_anomaly_matrix

# pylint: disable=C0103 # Does not conform to snake_case naming style


class ESMDABase(ABC):
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
    cov_dd: NDArrayFloat
        Autocovariance matrix of predicted data.
        Dimensions are (:math:`N_{obs}, N_{obs}`). if the matrix is diagonal, it is
        a 1D array.
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
    dd_correlation_matrix : Optional[csr_matrix]
        Correlation matrix based on spatial and temporal distances between
        observations and observations :math:`\rho_{DD}`. It is used to localize the
        autocovariance matrix of predicted data by applying an elementwise
        multiplication by this matrix.
        Expected dimensions are (:math:`N_{obs}`, :math:`N_{obs}`).
    md_correlation_matrix : Optional[csr_matrix]
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
    truncation: float
        A value in the range ]0, 1], used to determine the number of
        significant singular values kept when using svd for the inversion
        of $(C_{dd} + \alpha C_{d})$: Only the largest singular values are kept,
        corresponding to this fraction of the sum of the nonzero singular values.
        The goal of truncation is to deal with smaller matrices (dimensionality
        reduction), easier to inverse.
    """
    # pylint: disable=R0902 # Too many instance attributes
    __slots__: List[str] = [
        "obs",
        "_cov_obs",
        "cov_obs_cholesky",
        "d_obs_uc",
        "d_pred",
        "d_history",
        "m_prior",
        "_m_bounds",
        "m_history",
        "_inversion_type",
        "cov_md",
        "cov_dd",
        "forward_model",
        "forward_model_args",
        "forward_model_kwargs",
        "_n_assimilations",
        "_assimilation_step",
        "_dd_correlation_matrix",
        "_md_correlation_matrix",
        "save_ensembles_history",
        "rng",
        "is_forecast_for_last_assimilation",
        "batch_size",
        "is_parallel_analyse_step",
        "_truncation",
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
        dd_correlation_matrix: Optional[Union[NDArrayFloat, spmatrix]] = None,
        md_correlation_matrix: Optional[Union[NDArrayFloat, spmatrix]] = None,
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
            If a 1D array is passed, it represents a diagonal covariance matrix.
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
        inversion_type: Union[ESMDAInversionType, str] = ESMDAInversionType.NAIVE
            Type of inversion used to solve :math:`(C_DD + \alpha CD)^{-1)(d-dobs)`.
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
        """
        self.obs: NDArrayFloat = obs
        self.m_prior: NDArrayFloat = m_init
        self.save_ensembles_history: bool = save_ensembles_history
        self.m_history: list[NDArrayFloat] = []
        self.d_history: list[NDArrayFloat] = []
        self.d_pred: NDArrayFloat = np.zeros([self.d_dim, self.n_ensemble])
        self.cov_obs = cov_obs
        self.cov_md: NDArrayFloat = np.array([])
        self.cov_dd: NDArrayFloat = np.array([])
        self.forward_model: Callable[..., NDArrayFloat] = forward_model
        self.forward_model_args: Sequence[Any] = forward_model_args
        self.inversion_type = inversion_type  # type: ignore
        if forward_model_kwargs is None:
            forward_model_kwargs = {}
        self.forward_model_kwargs: Dict[str, Any] = forward_model_kwargs
        self._set_n_assimilations(n_assimilations)
        self._assimilation_step: int = 0
        self.dd_correlation_matrix = dd_correlation_matrix
        self.md_correlation_matrix = md_correlation_matrix
        self.m_bounds = m_bounds  # type: ignore
        if seed is not None:
            warnings.warn(
                DeprecationWarning(
                    "The keyword `seed` is now replaced by `random_state` "
                    "and will be dropped in 0.5.x."
                )
            )
        self.rng: np.random.RandomState = check_random_state(
            random_state
        )  # type: ignore
        self.is_forecast_for_last_assimilation: bool = is_forecast_for_last_assimilation
        self.batch_size = batch_size
        self.is_parallel_analyse_step: bool = is_parallel_analyse_step
        self.truncation = truncation

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
        return self.m_prior.shape[1]  # type: ignore

    @property
    def m_dim(self) -> int:
        """Return the length of the parameters vector."""
        return self.m_prior.shape[0]  # type: ignore

    @property
    def d_dim(self) -> int:
        """Return the number of forecast data."""
        return len(self.obs)

    @property
    def cov_obs(self) -> NDArrayFloat:
        """Get the observation errors covariance matrix."""
        return self._cov_obs

    @cov_obs.setter
    def cov_obs(self, cov: NDArrayFloat) -> None:
        """
        Set the observation errors covariance matrix.

        It must be a 2D array, or a 1D array if the covariance matrix is diagonal.
        """
        error = ValueError(
            "cov_obs must be a 2D matrix with "
            f"dimensions ({self.d_dim}, {self.d_dim})."
        )
        if len(cov.shape) > 2:
            raise error
        if cov.shape[0] != self.obs.size:  # type: ignore
            raise error
        if cov.ndim == 2:
            if cov.shape[0] != cov.shape[1]:  # type: ignore
                raise error

        # From iterative_ensemble_smoother code
        # Only compute the covariance factorization once
        # If it's a full matrix, we gain speedup by only computing cholesky once
        # If it's a diagonal, we gain speedup by never having to compute cholesky
        if cov.ndim == 2:
            self.cov_obs_cholesky: NDArrayFloat = sp.linalg.cholesky(cov, lower=False)
        else:
            self.cov_obs_cholesky = np.sqrt(cov)  # type: ignore

        self._cov_obs: NDArrayFloat = cov

    @property
    def anomalies(self) -> NDArrayFloat:
        r"""
        Return the matrix of anomalies.

        The anomaly matrix is defined as.


        Or in matrix form:

        .. math::

                \bm{A} = \bm{X}\left(\bm{I_{N_{e}}} - \dfrac{1}{N_{e}} \bm{11}^{T}
                \right) / \sqrt{N_{e}-1}.
        """
        return get_anomaly_matrix(self.m_prior)

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
        return self.anomalies @ self.anomalies.T

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
            self._m_bounds[:, 0] = -np.inf
            self._m_bounds[:, 1] = np.inf
        elif mb.shape[0] != self.m_dim:  # type: ignore
            raise ValueError(
                f"m_bounds is of shape {mb.shape} while it "
                f"should be of shape ({self.m_dim}, 2)"
            )
        else:
            self._m_bounds = mb

    @property
    def inversion_type(self) -> ESMDAInversionType:
        """Get the inversion_type."""
        return self._inversion_type

    @inversion_type.setter
    def inversion_type(self, inversion_type: Union[ESMDAInversionType, str]) -> None:
        """Set the inversion type."""
        if str(inversion_type) not in [v.value for v in ESMDAInversionType.to_list()]:
            raise ValueError(
                f"{str(inversion_type)} is not a supported inversion type! "
                f"Choose among {[v.value for v in ESMDAInversionType.to_list()]}"
            )
        self._inversion_type: ESMDAInversionType = ESMDAInversionType(
            str(inversion_type)
        )

    @property
    def dd_correlation_matrix(self) -> Optional[csr_matrix]:
        """Get the observations-observations localization matrix."""
        return self._dd_correlation_matrix

    @dd_correlation_matrix.setter
    def dd_correlation_matrix(
        self, mat: Optional[Union[NDArrayFloat, spmatrix]]
    ) -> None:
        """Set the observations-observations localization matrix."""
        if mat is None:
            self._dd_correlation_matrix = None
            return
        if (
            len(mat.shape) != 2
            or mat.shape[0] != mat.shape[1]
            or mat.shape[0] != self.d_dim
        ):
            raise ValueError(
                "dd_correlation_matrix must be a 2D square matrix with "
                f"dimensions ({self.d_dim}, {self.d_dim})."
            )
        self._dd_correlation_matrix = csr_matrix(mat)

    @property
    def md_correlation_matrix(self) -> Optional[csr_matrix]:
        """Get the parameters-observations localization matrix."""
        return self._md_correlation_matrix

    @md_correlation_matrix.setter
    def md_correlation_matrix(
        self, mat: Optional[Union[NDArrayFloat, spmatrix]]
    ) -> None:
        """Set the parameters-observations localization matrix."""
        if mat is None:
            self._md_correlation_matrix = None
            return
        if len(mat.shape) != 2:
            raise ValueError(
                "md_correlation_matrix must be a 2D matrix with "
                f"dimensions ({self.m_dim}, {self.d_dim})."
            )
        if mat.shape[0] != self.m_dim or mat.shape[1] != self.d_dim:
            raise ValueError(
                "md_correlation_matrix must be a 2D matrix with "
                f"dimensions ({self.m_dim}, {self.d_dim})."
            )
        self._md_correlation_matrix = csr_matrix(mat)

    @property
    def n_batches(self) -> int:
        """Number of batch used in the optimization."""
        return int(np.ceil(self.m_dim / self.batch_size))

    @property
    def truncation(self) -> float:
        """Return the truncation number for the svd in inversion."""
        return self._truncation

    @truncation.setter
    def truncation(self, truncation: float) -> None:
        """Return the truncation number for the svd in inversion."""
        if truncation > 1 or truncation <= 0:
            raise ValueError("The truncation number should be in ]0, 1]!")
        self._truncation = float(truncation)

    @abstractmethod
    def solve(self) -> None:
        """Solve the optimization problem with ES-MDA algorithm."""
        ...  # pragma: no cover

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

        Draw samples from zero-centered multivariate normal with cov=alpha * C_D,
        and add them to the observations. Notice that
        if C_D = L L.T by the cholesky factorization, then drawing y from
        a zero cented normal means that y := L @ z, where z ~ norm(0, 1)
        Therefore, scaling C_D by alpha is equivalent to scaling L with sqrt(alpha)

        """
        shape = (self.d_dim, self.n_ensemble)

        if self._cov_obs.ndim == 2:
            self.d_obs_uc = self.obs.reshape(-1, 1) + np.sqrt(
                inflation_factor
            ) * self.cov_obs_cholesky @ self.rng.normal(size=shape)
        else:
            self.d_obs_uc = self.obs.reshape(-1, 1) + np.sqrt(
                inflation_factor
            ) * self.rng.normal(size=shape) * self.cov_obs_cholesky.reshape(-1, 1)

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
        return self.m_prior + (
            inversion(
                self.inversion_type,  # type: ignore
                inflation_factor,
                self.cov_obs,
                self.cov_obs_cholesky,
                self.d_obs_uc,
                self.d_pred,
                self.m_prior,
                dd_corr_mat=self.dd_correlation_matrix,
                md_corr_mat=self.md_correlation_matrix,
                truncation=self.truncation,
            )
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
                        index * self.batch_size,
                        max([(index + 1) * self.batch_size, self.m_dim]),
                    )
                    m_pred[_slice, :] = res
        else:
            for index in range(self.n_batches):
                _slice = slice(
                    index * self.batch_size,
                    max([(index + 1) * self.batch_size, self.m_dim]),
                )
                m_pred[_slice, :] = self._get_batch_m_update(index, inflation_factor)

        return m_pred

    def _get_batch_m_update(self, index: int, inflation_factor: float) -> NDArrayFloat:
        _slice = slice(
            index * self.batch_size, max([(index + 1) * self.batch_size, self.m_dim])
        )

        if self.md_correlation_matrix is not None:
            batch_cov_md: Optional[csr_matrix] = self.md_correlation_matrix[_slice, :]
        else:
            batch_cov_md = None

        return self.m_prior[_slice, :] + (
            inversion(
                self.inversion_type,  # type: ignore
                inflation_factor,
                self.cov_obs,
                self.cov_obs_cholesky,
                self.d_obs_uc,
                self.d_pred,
                self.m_prior[_slice, :].reshape(
                    -1, self.m_prior.shape[-1]
                ),  # ensure 2d array
                dd_corr_mat=self.dd_correlation_matrix,
                md_corr_mat=batch_cov_md,
                truncation=1.0,
            )
        )

    def _apply_bounds(self, m_pred: NDArrayFloat) -> NDArrayFloat:
        """Apply bounds constraints to the adjusted parameters."""
        return np.clip(m_pred.T, self.m_bounds[:, 0], self.m_bounds[:, 1]).T
