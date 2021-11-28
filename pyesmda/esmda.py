"""
Implement the ES-MDA algorithms.

@author: acollet
"""
from typing import List, Union, Callable
import numpy as np


class ESMDA():
    """
    Ensemble Smoother with Multiple Data Assimilations.

    Implement the ES-MDA as proposed by  Emerick, A. A. and A. C. Reynolds
    [1]_, [2]_.

    Parameters
    ----------
    dobs : np.array
        Obsevrations vector.
    m_init : np.array
        Initial ensemble of N_{e} parameters vector.
    stdev_d: np.array
        Standard deviation of the observations.
    stdev_m: np.array
        Standard deviation of the parameters.
    forward_model: callable
        Function calling the non-linear observation model (forward model)
        for all ensemble members and returning the predicted data for
        each ensemble member.
    forward_model_args: tuple
        Additional args for the callable forward_model.
    forward_model_kwargs: dict
        Additional kwargs for the callable forward_model.
    n_assimilation : int, optional
        Number of data assimilations. The default is 4.
    alpha : Union[List[int], None], optional
        Multiplication factor used to inflate the covariance matrix of the
        measurement errors. The default is None.
    m_bounds : Union[List[int], None], optional
        Top and bottom bounds on the initial ensemble of N_{e} parameters
        vector. The default is None.

    References
    ----------
    .. [1] Emerick, A. A. and A. C. Reynolds, Ensemble smoother with multiple
        data assimilation, Computers & Geosciences, 2012.
    .. [2] Emerick, A. A. and A. C. Reynolds. (2013). History-Matching
        Production and Seismic Data in a Real Field Case Using the Ensemble
        Smoother With Multiple Data Assimilation. Society of Petroleum
        Engineers - SPE Reservoir Simulation Symposium
        2013. 2. 10.2118/163675-MS.
    """

    def __init__(self,
                 obs: np.array,
                 m_init: np.array,
                 stdev_d: np.array,
                 stdev_m: np.array,
                 forward_model: Callable,
                 forward_model_args: tuple = (),
                 forward_model_kwargs: dict = {},
                 n_assimilation: int = 4,
                 alpha: Union[List[int], None] = None,
                 m_bounds: Union[np.array, None] = None):
        """Construct method."""
        self.dobs = obs
        self.m_prior = m_init
        # List of average m values at the end of each assimilation
        self.m_mean = []
        # Initialize d_piror with observation
        # Must be initialized after m_prior because it is used in the
        # @property
        self.d_prior = np.array([obs] * self.n_ensemble)
        self.d_pred = np.zeros([self.n_ensemble, self.d_dim])
        self.stdev_d = stdev_d
        self.stdev_m = stdev_m
        self.forward_model = forward_model
        self.forward_model_args = forward_model_args
        self.forward_model_kwargs = forward_model_kwargs
        self.n_assimilation = n_assimilation
        self.alpha = alpha
        self.m_bounds = m_bounds

    @property
    def n_assimilation(self):
        """Return the number of assimilation to perfom."""
        return self._n_assimilation

    @n_assimilation.setter
    def n_assimilation(self, n):
        """Set the number of assimilation to perfom."""
        if type(n) != int:
            raise TypeError("The number of assimilation must be an interger.")
        elif n < 1:
            raise ValueError("The number of assimilation must be 1 or more.")
        self._n_assimilation = n

    @property
    def n_ensemble(self):
        """Return the number of ensemble members."""
        return self.m_prior.shape[0]

    @property
    def m_dim(self):
        """Return the length of the parameters vector."""
        return self.m_prior.shape[1]

    @property
    def d_dim(self):
        """Return the number of forecast data."""
        return len(self.dobs)

    @property
    def stdev_d(self):
        """Get the observation errors covariance matrix."""
        return self._stdev_d

    @stdev_d.setter
    def stdev_d(self, s):
        """Set the observation errors covariance matrix."""
        if s.shape[0] != s.shape[1]:
            raise ValueError("stdev must be a square matrix with same "
                             "dimensions as observations vector.")
        elif s.shape[0] != self.d_dim:
            raise ValueError("stdev_s must be a square matrix with same "
                             "dimension as observation vector.")
        else:
            self._stdev_d = s

    @property
    def stdev_m(self):
        """Get the parameter errors covariance matrix."""
        return self._stdev_m

    @stdev_m.setter
    def stdev_m(self, s):
        """Set the parameter errors covariance matrix."""
        if s.shape[0] != self.m_dim:
            raise ValueError("stdev_m must be of the same "
                             "dimension as the parameter vector.")
        else:
            self._stdev_m = s

    @property
    def m_bounds(self):
        """Get the parameter errors covariance matrix."""
        return self._m_bounds

    @m_bounds.setter
    def m_bounds(self, mb):
        """Set the parameter errors covariance matrix."""
        if mb is None:
            # In that case, create an array of nan.
            self._m_bounds = np.empty([self.m_dim, 2])
            self._m_bounds[:] = np.nan
        elif mb.shape[0] != self.m_dim:
            raise ValueError(f"m_bounds is of size {mb.shape} while it"
                             f"should be of size ({self.m_dim} x 2)")
        else:
            self._m_bounds = mb

    @property
    def alpha(self):
        r"""
        Get the alpha coefficients used by ES-MDA.

        Single and multiple data assimilation are equivalent for the
        linear-Gaussian case as long as the factor :math:`\alpha_{l}` used to
        inflate the covariance matrix of the measurement errors satisfy the
        following condition:

        .. math::
            \sum_{l=1}^{N_{a}} \frac{1}{\alpha_{l}} = 1

        In practise, :math:`\alpha_{l} = N_{a}` is a good choice [1]_.
        """
        return self._alpha

    @alpha.setter
    def alpha(self, a):
        """Set the alpha coefficients used by ES-MDA."""
        if a is None:
            self._alpha = np.array([self.n_assimilation] * self.n_assimilation)
        elif len(a) != self.n_assimilation:
            raise ValueError("The length of alpha should match n_assimilation")
        else:
            self._alpha = a

    def solve(self):
        """Solve the optimization problem with ES-MDA algorithm."""
        for assimilation_iteration in range(self.n_assimilation):
            print(f"Assimilation # {assimilation_iteration + 1}")
            self.forecast()
            self.pertrub(assimilation_iteration)
            self.approximate_covariance_matrices()
            self.analyse(assimilation_iteration)

    def forecast(self):
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

        Returns
        -------
        None.

        """
        self.d_pred = self.forward_model(self.m_prior,
                                         *self.forward_model_args,
                                         **self.forward_model_kwargs)

    def pertrub(self, assimilation_iteration):
        r"""
        Perturbation of the observation vector step of ES-MDA.

        Perturb the vector of observations

        .. math::
            d^{l}_{uc,j} = d_{obs} + \sqrt{\alpha_{l+1}}C_{D}^{1/2}Z_{d},
            \textrm{for } j=1,2,...,N_{e},

        where :math:`Z_{d} \sim \mathcal{N}(O, I_{N_{d}})`.

        Returns
        -------
        None.

        """
        self.d_obs_uc = np.zeros([self.n_ensemble, self.d_dim])
        for i in range(self.d_dim):
            self.d_obs_uc[:, i] = (
                self.dobs[i]
                + np.sqrt(self.alpha[assimilation_iteration])
                * np.random.normal(0, np.abs(self.stdev_d[i, i]),
                                   self.n_ensemble
                                   )
                )

    def approximate_covariance_matrices(self):
        """
        Calculate Average and Covariance MD and Covariance DD.

        The covariance matrices :math:`C^{l}_{MD}` and :math:`C^{l}_{DD}`
        are approximated from the ensemble in the standard way of EnKF
        [3]_, [4]_.

        References
        ----------
        .. [3] Evensen, G., Data Assimilation: The Ensemble Kalman Filter,
            Springer, Berlin, 2007
        .. [4] Aanonsen, S. I., G. Nævdal, D. S. Oliver, A. C. Reynolds, and
            B. Valles, Review of ensemble Kalman filter in petroleum
            engineering, SPE Journal, 14(3), 393–412, 2009.
        """
        # Average of parameters and predictions of the ensemble members
        m_average = np.mean(self.m_prior, axis=0)
        d_average = np.mean(self.d_pred, axis=0)
        # Delta with average per ensemble member
        delta_m = (self.m_prior - m_average)
        delta_d = (self.d_pred - d_average)

        dd_md = 0.0
        dd_dd = 0.0

        for j in range(self.n_ensemble):
            dd_md += np.outer(delta_m[j, :], delta_d[j, :])
            dd_dd += np.outer(delta_d[j, :], delta_d[j, :])

        self.cov_md = dd_md / (self.n_ensemble - 1.0)
        self.cov_dd = dd_dd / (self.n_ensemble - 1.0)

    def analyse(self, assimilation_iteration):
        r"""
        Analysis step of the ES-MDA.

        Update the vector of model parameters using

        .. math::
           m^{l+1}_{j} = m^{l}_{j} + C^{l}_{MD}\left(C^{l}_{DD}+\alpha_{l+1}
           C_{D}\right)^{-1} \left(d^{l}_{uc,j} - d^{l}_{j} \right),
           \textrm{for } j=1,2,...,N_{e}.

        Returns
        -------
        None.
        """
        # predicted parameters
        m_pred = np.zeros([self.n_ensemble, self.m_dim])
        for j in range(self.n_ensemble):
            tmp_mat = np.matmul(
                self.cov_md, np.linalg.inv(self.cov_dd
                                           + self.alpha[assimilation_iteration]
                                           * self.stdev_d)
            )
            tmp_vec = self.d_obs_uc[j, :] - self.d_pred[j, :]
            m_pred[j, :] = self.m_prior[j, :] + np.matmul(tmp_mat, tmp_vec)

        # Apply bounds constraints to parameters
        m_pred = np.where(m_pred < self.m_bounds[:, 0],
                          self.m_bounds[:, 0], m_pred)  # lower bounds
        m_pred = np.where(m_pred > self.m_bounds[:, 1],
                          self.m_bounds[:, 1], m_pred)  # upper bounds

        # Update the prior parameter for next iteration
        self.m_prior = m_pred

        # Plotting for change of average of the parameters
        self.m_mean.append(np.average(m_pred, axis=0))
