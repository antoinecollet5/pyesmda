"""
Implement the ES-MDA algorithms.

@author: acollet
"""
from typing import List, Union, Callable, Dict, Sequence, Any, Optional
import numpy as np
import numpy.typing as npt


class ESMDA:
    """
    Ensemble Smoother with Multiple Data Assimilations.

    Implement the ES-MDA as proposed by  Emerick, A. A. and A. C. Reynolds
    :cite:p:`emerickEnsembleSmootherMultiple2013,
    emerickHistoryMatchingProductionSeismic2013`.

    Attributes
    ----------
    d_dim : int
        Number of observation values :math:`N_{obs}`, and consequently of
        predicted values.
    obs : npt.NDArray[np.float64]
        Obsevrations vector with dimensions (:math:`N_{obs}`).
    cov_d: npt.NDArray[np.float64]
        Covariance matrix of observed data measurement errors with dimensions
        (:math:`N_{obs}`, :math:`N_{obs}`).
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
        Dimensions are (:math:`N_{m}, N_{obs}`)
    cov_dd: npt.NDArray[np.float64]
        Autocovariance matrix of predicted data.
        Dimensions are (:math:`N_{obs}, N_{obs}`)
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
    alpha : List[float]
        List of multiplication factor used to inflate the covariance matrix of the
        measurement errors. Dimensions
    save_ensembles_history: bool
        Whether to save the history predictions and parameters over the assimilations.
    """

    __slots__: List[str] = [
        "obs",
        "_cov_d",
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
        "_alpha",
        "save_ensembles_history",
    ]

    def __init__(
        self,
        obs: npt.NDArray[np.float64],
        m_init: npt.NDArray[np.float64],
        cov_d: npt.NDArray[np.float64],
        forward_model: Callable[..., npt.NDArray[np.float64]],
        forward_model_args: Sequence[Any] = (),
        forward_model_kwargs: Optional[Dict[str, Any]] = None,
        n_assimilations: int = 4,
        alpha: Optional[Sequence[float]] = None,
        m_bounds: Optional[npt.NDArray[np.float64]] = None,
        save_ensembles_history: bool = False,
    ) -> None:
        """Construct the instance.

        Parameters
        ----------
        obs : npt.NDArray[np.float64]
            Obsevrations vector with dimension :math:`N_{obs}`.
        m_init : npt.NDArray[np.float64]
            Initial ensemble of parameters vector with dimensions
            (:math:`N_{e}`, :math:`N_{m}`).
        cov_d: npt.NDArray[np.float64]
            Covariance matrix of observed data measurement errors with dimensions
            (:math:`N_{obs}`, :math:`N_{obs}`).
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
        alpha : Optional[Sequence[float]]
            Multiplication factor used to inflate the covariance matrix of the
            measurement errors. The default is None.
        m_bounds : Optional[npt.NDArray[np.float64]], optional
            Lower and upper bounds for the :math:`N_{m}` parameter values.
            Expected dimensions are (:math:`N_{m}`, 2) with lower bounds on the first
            column and upper on the second one. The default is None.
        save_ensembles_history: bool, optional
            Whether to save the history predictions and parameters over
            the assimilations. The default is False.
        """
        self.obs: npt.NDArray[np.float64] = obs
        self.m_prior: npt.NDArray[np.float64] = m_init
        self.save_ensembles_history: bool = save_ensembles_history
        self.m_history: list[npt.NDArray[np.float64]] = []
        self.d_history: list[npt.NDArray[np.float64]] = []
        self.d_pred: npt.NDArray[np.float64] = np.zeros([self.n_ensemble, self.d_dim])
        self.cov_d = cov_d
        self.d_obs_uc: npt.NDArray[np.float64] = np.array([])
        self.cov_md: npt.NDArray[np.float64] = np.array([])
        self.cov_dd: npt.NDArray[np.float64] = np.array([])
        self.forward_model: Callable = forward_model
        self.forward_model_args: Sequence[Any] = forward_model_args
        if forward_model_kwargs is None:
            forward_model_kwargs: Dict[str, Any] = {}
        self.forward_model_kwargs: Dict[str, Any] = forward_model_kwargs
        self.n_assimilations = n_assimilations
        self.alpha = alpha
        self.m_bounds = m_bounds

    @property
    def n_assimilations(self) -> int:
        """Return the number of assimilations to perfom."""
        return self._n_assimilations

    @n_assimilations.setter
    def n_assimilations(self, n: int) -> None:
        """Set the number of assimilations to perfom."""
        if not isinstance(n, int):
            raise TypeError("The number of assimilations must be a positive interger.")
        if n < 1:
            raise ValueError("The number of assimilations must be 1 or more.")
        self._n_assimilations = n

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
    def cov_d(self) -> npt.NDArray[np.float64]:
        """Get the observation errors covariance matrix."""
        return self._cov_d

    @cov_d.setter
    def cov_d(self, s: npt.NDArray[np.float64]) -> None:
        """Set the observation errors covariance matrix."""
        if len(s.shape) != 2 or s.shape[0] != s.shape[1]:
            raise ValueError(
                "cov_d must be a square matrix with same "
                "dimensions as the observations vector."
            )
        if s.shape[0] != self.d_dim:
            raise ValueError(
                "cov_d must be a square matrix with same "
                "dimensions as the observations vector."
            )
        self._cov_d = s

    @property
    def m_bounds(self) -> npt.NDArray[np.float64]:
        """Get the parameter errors covariance matrix."""
        return self._m_bounds

    @m_bounds.setter
    def m_bounds(self, mb: npt.NDArray[np.float64]) -> None:
        """Set the parameter errors covariance matrix."""
        if mb is None:
            # In that case, create an array of nan.
            self._m_bounds: npt.NDArray[np.float64] = np.empty(
                [self.m_dim, 2], dtype=np.float64
            )
            self._m_bounds[:] = np.nan
        elif mb.shape[0] != self.m_dim:
            raise ValueError(
                f"m_bounds is of size {mb.shape} while it "
                f"should be of size ({self.m_dim}, 2)"
            )
        else:
            self._m_bounds = mb

    @property
    def alpha(self) -> Union[List[float], npt.NDArray[np.float64]]:
        r"""
        Get the alpha coefficients used by ES-MDA.

        Single and multiple data assimilation are equivalent for the
        linear-Gaussian case as long as the factor :math:`\alpha_{l}` used to
        inflate the covariance matrix of the measurement errors satisfy the
        following condition:

        .. math::
            \sum_{l=1}^{N_{a}} \frac{1}{\alpha_{l}} = 1

        In practise, :math:`\alpha_{l} = N_{a}` is a good choice
        :cite:p:`emerickEnsembleSmootherMultiple2013`.
        """
        return self._alpha

    @alpha.setter
    def alpha(self, a: Sequence[float]) -> None:
        """Set the alpha coefficients used by ES-MDA."""
        if a is None:
            self._alpha: npt.NDArray[np.float64] = np.array(
                [1 / self.n_assimilations] * self.n_assimilations, dtype=np.float64
            )
        elif len(a) != self.n_assimilations:
            raise ValueError("The length of alpha should match n_assimilations")
        else:
            self._alpha = np.array(a)

    def solve(self) -> None:
        """Solve the optimization problem with ES-MDA algorithm."""
        if self.save_ensembles_history:
            self.m_history.append(self.m_prior)  # save m_init
        for assimilation_iteration in range(self.n_assimilations):
            print(f"Assimilation # {assimilation_iteration + 1}")
            self._forecast()
            self._pertrub(assimilation_iteration)
            self._approximate_covariance_matrices()
            self._analyse(assimilation_iteration)

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

    def _pertrub(self, assimilation_iteration: int) -> None:
        r"""
        Perturbation of the observation vector step of ES-MDA.

        Perturb the vector of observations

        .. math::
            d^{l}_{uc,j} = d_{obs} + \sqrt{\alpha_{l+1}}C_{D}^{1/2}Z_{d},
            \textrm{for } j=1,2,...,N_{e},

        where :math:`Z_{d} \sim \mathcal{N}(O, I_{N_{d}})`.
        """
        self.d_obs_uc = np.zeros([self.n_ensemble, self.d_dim])
        for i in range(self.d_dim):
            self.d_obs_uc[:, i] = self.obs[i] + np.sqrt(
                self.alpha[assimilation_iteration]
            ) * np.random.normal(0, np.abs(self.cov_d[i, i]), self.n_ensemble)

    def _approximate_covariance_matrices(self) -> None:
        """
        Calculate Average and Covariance MD and Covariance DD.

        The covariance matrices :math:`C^{l}_{MD}` and :math:`C^{l}_{DD}`
        are approximated from the ensemble in the standard way of EnKF
        :cite:p:`evensenDataAssimilationEnsemble2007,aanonsenEnsembleKalmanFilter2009`.
        """
        # Average of parameters and predictions of the ensemble members
        m_average = np.mean(self.m_prior, axis=0)
        d_average = np.mean(self.d_pred, axis=0)
        # Delta with average per ensemble member
        delta_m = self.m_prior - m_average
        delta_d = self.d_pred - d_average

        dd_md = 0.0
        dd_dd = 0.0

        for j in range(self.n_ensemble):
            dd_md += np.outer(delta_m[j, :], delta_d[j, :])
            dd_dd += np.outer(delta_d[j, :], delta_d[j, :])

        self.cov_md = dd_md / (self.n_ensemble - 1.0)
        self.cov_dd = dd_dd / (self.n_ensemble - 1.0)

    def _analyse(self, assimilation_iteration: int) -> None:
        r"""
        Analysis step of the ES-MDA.

        Update the vector of model parameters using

        .. math::
           m^{l+1}_{j} = m^{l}_{j} + C^{l}_{MD}\left(C^{l}_{DD}+\alpha_{l+1}
           C_{D}\right)^{-1} \left(d^{l}_{uc,j} - d^{l}_{j} \right),
           \textrm{for } j=1,2,...,N_{e}.
        """
        # predicted parameters
        m_pred = np.zeros([self.n_ensemble, self.m_dim])
        for j in range(self.n_ensemble):
            tmp_mat = np.matmul(
                self.cov_md,
                np.linalg.inv(
                    self.cov_dd + self.alpha[assimilation_iteration] * self.cov_d
                ),
            )
            tmp_vec = self.d_obs_uc[j, :] - self.d_pred[j, :]
            m_pred[j, :] = self.m_prior[j, :] + np.matmul(tmp_mat, tmp_vec)

        # Apply bounds constraints to parameters
        m_pred = np.where(
            m_pred < self.m_bounds[:, 0], self.m_bounds[:, 0], m_pred
        )  # lower bounds
        m_pred = np.where(
            m_pred > self.m_bounds[:, 1], self.m_bounds[:, 1], m_pred
        )  # upper bounds

        # Update the prior parameter for next iteration
        self.m_prior = m_pred

        # Saving the parameters history
        if self.save_ensembles_history:
            self.m_history.append(m_pred)
