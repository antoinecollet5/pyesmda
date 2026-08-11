# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021-2026 Antoine COLLET

"""
Implement a base class for the ES-MDA algorithms and variants.

@author: acollet
"""

import logging
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import covmats
import numpy as np
from joblib import Parallel, delayed
from scipy._lib._util import check_random_state

from pyesmda._inversion import (
    ESMDAInversionType,
    _check_localization_inversion_compatibility,
    _run_batch_update,
    inversion,
)
from pyesmda._localization import LocalizationStrategy, NoLocalization
from pyesmda._utils import (
    NDArrayFloat,
    NDArrayInt,
    check_nans_in_predictions,
    get_anomaly_matrix,
    get_failed_members_indices,
    inflate_ensemble_around_its_mean,
)


class ESMDABase(ABC):
    r"""
    Ensemble Smoother with Multiple Data Assimilation.

    Implement the ES-MDA as proposed by  Emerick, A. A. and A. C. Reynolds
    :cite:p:`emerickEnsembleSmootherMultiple2013,
    emerickHistoryMatchingProductionSeismic2013`.
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
        "_inversion_type",
        "cov_md",
        "cov_dd",
        "forward_model",
        "forward_model_args",
        "forward_model_kwargs",
        "_n_assimilations",
        "_assimilation_step",
        "_C_DD_localization",
        "_C_MD_localization",
        "save_ensembles_history",
        "rng",
        "is_forecast_for_last_assimilation",
        "batch_size",
        "is_parallel_analyse_step",
        "_truncation",
        "logger",
        "_max_failure_fraction",
        "_initial_n_ensemble",
        "_active_member_indices",
        "_excluded_member_indices",
    ]

    def __init__(
        self,
        obs: NDArrayFloat,
        m_init: NDArrayFloat,
        cov_obs: covmats.CovarianceMatrix,
        forward_model: Callable[..., NDArrayFloat],
        forward_model_args: Sequence[Any] = (),
        forward_model_kwargs: Optional[Dict[str, Any]] = None,
        n_assimilations: int = 4,
        inversion_type: Union[
            ESMDAInversionType, str
        ] = ESMDAInversionType.SUBSPACE_RESCALED,
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
        max_failure_fraction: float = 0.0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        # pylint: disable=R0913 # Too many arguments
        # pylint: disable=R0914 # Too many local variables
        """Initialize the instance."""
        self.obs: NDArrayFloat = obs
        r"""Obsevrations vector with dimensions (:math:`N_{\mathrm{obs}}`)."""

        self.m_prior: NDArrayFloat = m_init
        r"""
        Vectors of parameter values (one vector for each ensemble member) used in the
        last assimilation step; Dimensions are (:math:`N_{m}`, :math:`N_{e}`).
        """

        self._initial_n_ensemble: int = self.n_ensemble
        """Number of ensemble members at initialization, before any exclusion."""

        self._active_member_indices: NDArrayInt = np.arange(self._initial_n_ensemble)
        """
        Indices, in the original (initial) ensemble, of the members that are
        still active, i.e., that have not been excluded because of a forward
        model failure.
        """

        self._excluded_member_indices: List[int] = []
        """
        Indices, in the original (initial) ensemble, of the members that have
        been excluded so far because of a forward model failure.
        """

        self.max_failure_fraction = max_failure_fraction

        self.save_ensembles_history: bool = save_ensembles_history
        """
        Whether to save the history predictions and parameters over the assimilations.
        """

        self.m_history: list[NDArrayFloat] = []
        """List of successive :py:attr:`m_prior`."""

        self.d_history: list[NDArrayFloat] = []
        """List of vectors of predicted values obtained at each assimilation step."""

        self.d_pred: NDArrayFloat = np.zeros([self.d_dim, self.n_ensemble])
        """Vectors of predicted values (one for each ensemble member)
        with dimensions (:math:`N_{obs}`, :math:`N_{e}`).
        """

        self.cov_obs = cov_obs
        self.cov_md: NDArrayFloat = np.array([])
        """
        Cross-covariance matrix between the forecast state vector and predicted data;
        Dimensions are (:math:`N_{m}, N_{obs}`).
        """

        self.cov_dd: NDArrayFloat = np.array([])
        """
        Autocovariance matrix of estimated parameters;
        Dimensions are (:math:`N_{m}, N_{m}`).
        """

        self.forward_model: Callable[..., NDArrayFloat] = forward_model
        """
        Function calling the non-linear observation model (forward model)
        for all ensemble members and returning the predicted data for
        each ensemble member.
        """

        self.forward_model_args: Sequence[Any] = forward_model_args
        """Additional args for the callable forward_model."""

        self.inversion_type = inversion_type

        if forward_model_kwargs is None:
            forward_model_kwargs = {}
        self.forward_model_kwargs: Dict[str, Any] = forward_model_kwargs
        """
        Function calling the non-linear observation model (forward model)
        for all ensemble members and returning the predicted data for
        each ensemble member.
        """

        self._set_n_assimilations(n_assimilations)
        self._assimilation_step: int = 0

        self.C_DD_localization = C_DD_localization
        self.C_MD_localization = C_MD_localization

        self.m_bounds = m_bounds
        if seed is not None:
            warnings.warn(
                DeprecationWarning(
                    "The keyword `seed` is now replaced by `random_state` "
                    "and has been dropped since version 0.4.3."
                )
            )
        self.rng: np.random.RandomState = check_random_state(random_state)
        """The random number generator used in the predictions perturbation step."""

        self.is_forecast_for_last_assimilation: bool = is_forecast_for_last_assimilation
        """
        Whether to compute the predictions for the ensemble obtained at the
        last assimilation step.
        """

        self.batch_size = batch_size
        """
        Number of parameters that are assimilated at once; This option is
        available to overcome memory limitations when the number of parameters is
        large; In that case, the size of the covariance matrices tends to explode
        and the update step must be performed by chunks of parameters.
        """

        self.is_parallel_analyse_step: bool = is_parallel_analyse_step
        """
        Whether to use parallel computing for the analyse step if the number of
        batch is above one.
        """

        self.truncation: float = truncation
        """
        A value in the range ]0, 1], used to determine the number of
        significant singular values kept when using svd for the inversion
        of $(C_{dd} + \alpha C_{d})$: Only the largest singular values are kept,
        corresponding to this fraction of the sum of the nonzero singular values;
        The goal of truncation is to deal with smaller matrices (dimensionality
        reduction), easier to inverse.
        """

        # Inflate the initial ensemble respecting the given bounds
        if cov_mm_inflation_factor != 1.0:
            self.m_prior = self._apply_bounds(
                inflate_ensemble_around_its_mean(
                    m_init, inflation_factor=cov_mm_inflation_factor
                )
            )
        self.logger: Optional[logging.Logger] = logger
        """Optional :py::class:`logging.Logger` instance used for event logging."""

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
        return self.m_prior.shape[1]

    @property
    def m_dim(self) -> int:
        """Return the length of the parameters vector."""
        return self.m_prior.shape[0]

    @property
    def d_dim(self) -> int:
        """Return the number of forecast data."""
        return len(self.obs)

    @property
    def cov_obs(self) -> covmats.CovarianceMatrix:
        """Get the observation errors covariance matrix."""
        return self._cov_obs

    @cov_obs.setter
    def cov_obs(self, cov: covmats.CovarianceMatrix) -> None:
        """
        Set the observation errors covariance matrix.

        It must be a 2D array, or a 1D array if the covariance matrix is diagonal.
        """
        error = ValueError(
            "`cov_obs` must be an implementation of `covmats.CovarianceMatrix`"
            f" with dimensions ({self.d_dim}, {self.d_dim})."
        )
        if not isinstance(cov, covmats.CovarianceMatrix):
            raise error
        if cov.shape != (self.obs.size, self.obs.size):
            raise error

        self._cov_obs: covmats.CovarianceMatrix = cov

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
        elif mb.shape[0] != self.m_dim:
            raise ValueError(
                f"m_bounds is of shape {mb.shape} while it "
                f"should be of shape ({self.m_dim}, 2)"
            )
        else:
            self._m_bounds = mb

    @property
    def inversion_type(self) -> ESMDAInversionType:
        """Inversion type. See :py:class:`ESMDAInversionType` for more details."""
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
        _check_localization_inversion_compatibility(
            self.inversion_type, self.C_DD_localization
        )

    @property
    def C_DD_localization(self) -> LocalizationStrategy:
        r"""
        Localization operator :math:`\rho_{DD}` applied to the predictions
        empirical auto-covariance matrices; Expected dimensions of the operator are
        (:math:`N_{obs}`, :math:`N_{obs}`); It can be fixed (defined correlation
        matrix used for all iterations) or adaptive and even user defined;
        See implementations of :py:class:`pyesmda.LocalizationStrategy`.
        """
        if not hasattr(self, "_CDD_localization"):
            return NoLocalization()
        return self._C_DD_localization

    @C_DD_localization.setter
    def C_DD_localization(self, C_DD_localization: LocalizationStrategy) -> None:
        """Set the inversion type."""
        C_DD_localization.check_localization_shape(
            (self.d_dim, self.d_dim), "C_DD_localization"
        )
        _check_localization_inversion_compatibility(
            self.inversion_type, C_DD_localization
        )
        self._C_DD_localization: LocalizationStrategy = C_DD_localization

    @property
    def C_MD_localization(self) -> LocalizationStrategy:
        r"""
        Localization operator :math:`\rho_{DD}` applied to the parameters-predictions
        empirical corss-covariance matrices; Expected dimensions of the operator are
        (:math:`N_{m}`, :math:`N_{obs}`); It can be fixed (defined correlation
        matrix used for all iterations) or adaptive and even user defined;
        See implementations of :py:class:`pyesmda.LocalizationStrategy`.
        """
        return self._C_MD_localization

    @C_MD_localization.setter
    def C_MD_localization(self, C_MD_localization: LocalizationStrategy) -> None:
        """Set the inversion type."""
        C_MD_localization.check_localization_shape(
            (self.m_dim, self.d_dim), "C_MD_localization"
        )
        self._C_MD_localization: LocalizationStrategy = C_MD_localization

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

    @property
    def max_failure_fraction(self) -> float:
        """
        Get the maximum fraction of the initial ensemble allowed to fail.

        A "failed" member is one for which the forward model returned at least
        one NaN value (typically because of a non-convergence). Failed members
        are excluded from the ensemble. If the cumulative fraction of failed
        members exceeds this threshold, an exception is raised. A value of 0.0
        means that no failure at all is tolerated.
        """
        return self._max_failure_fraction

    @max_failure_fraction.setter
    def max_failure_fraction(self, max_failure_fraction: float) -> None:
        """Set the maximum fraction of the initial ensemble allowed to fail."""
        if not 0.0 <= max_failure_fraction < 1.0:
            raise ValueError(
                f"max_failure_fraction should be in [0, 1[! Got {max_failure_fraction}."
            )
        self._max_failure_fraction = float(max_failure_fraction)

    @property
    def active_member_indices(self) -> NDArrayFloat:
        """
        Get the indices, in the original (initial) ensemble, of the currently
        active members, i.e., the members that have not been excluded because
        of a forward model failure. Read-only.
        """
        return self._active_member_indices

    @property
    def excluded_member_indices(self) -> List[int]:
        """
        Get the indices, in the original (initial) ensemble, of the members
        that have been excluded so far because of a forward model failure.
        Read-only.
        """
        return list(self._excluded_member_indices)

    @property
    def n_excluded_members(self) -> int:
        """Get the number of ensemble members excluded so far. Read-only."""
        return len(self._excluded_member_indices)

    @property
    def failure_fraction(self) -> float:
        """
        Get the cumulative fraction of the initial ensemble that has failed
        so far. Read-only.
        """
        return self.n_excluded_members / self._initial_n_ensemble

    def loginfo(self, msg: str) -> None:
        """Log the message."""
        if self.logger is not None:
            self.logger.info(msg)

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

        # Handle members for which the forward model failed (NaN predictions):
        # either exclude them (within the allowed failure fraction) or raise.
        self._handle_failed_members()

        if self.save_ensembles_history:
            self.d_history.append(self.d_pred)

    def _handle_failed_members(self) -> None:
        """
        Detect ensemble members for which the forward model failed and exclude them.

        A member is considered failed if its prediction vector contains at least
        one NaN value (typically because the forward/reservoir simulation did not
        converge). If ``max_failure_fraction`` is 0.0 (the default), any failure
        immediately raises an exception (the historical, strict behavior). Otherwise,
        failed members are dropped from :py:attr:`m_prior` and :py:attr:`d_pred` (and
        thus excluded from the analysis/inversion step and from subsequent
        assimilations), as long as the cumulative fraction of failed members
        (relative to the initial ensemble size) does not exceed
        ``max_failure_fraction``. If it does, an exception is raised.
        """
        local_failed_indices = get_failed_members_indices(self.d_pred)
        if local_failed_indices.size == 0:
            return

        if self.max_failure_fraction == 0.0:
            # Historical strict behavior: no failure tolerated at all.
            check_nans_in_predictions(self.d_pred, self._assimilation_step)

        # Map the (local) failed column indices back to indices in the
        # original, initial ensemble.
        new_failed_original_indices = self._active_member_indices[local_failed_indices]

        total_n_failed = self.n_excluded_members + local_failed_indices.size
        new_failure_fraction = total_n_failed / self._initial_n_ensemble

        if new_failure_fraction > self.max_failure_fraction:
            raise Exception(
                f"Something went wrong after assimilation step "
                f"{self._assimilation_step} -> NaN values are found in "
                "predictions for members "
                f"{[int(i) for i in new_failed_original_indices]} ! "
                f"This brings the cumulative failure fraction to "
                f"{new_failure_fraction:.2%}, which exceeds the allowed "
                f"max_failure_fraction of {self.max_failure_fraction:.2%}."
            )

        # Exclude the failed members: keep only the active (non-failed) columns.
        active_mask = np.ones(self.n_ensemble, dtype=bool)
        active_mask[local_failed_indices] = False

        self.loginfo(
            f"Excluding {local_failed_indices.size} failed ensemble member(s) "
            f"(original indices {[int(i) for i in new_failed_original_indices]}) "
            f"at assimilation step {self._assimilation_step}. Cumulative failure "
            f"fraction: {new_failure_fraction:.2%} "
            f"(max allowed: {self.max_failure_fraction:.2%})."
        )

        self._excluded_member_indices.extend(
            int(i) for i in new_failed_original_indices
        )
        self._active_member_indices = self._active_member_indices[active_mask]
        self.m_prior = self.m_prior[:, active_mask]
        self.d_pred = self.d_pred[:, active_mask]

        if self.n_ensemble < 2:
            raise Exception(
                "Too many ensemble members have failed: fewer than 2 members "
                "remain, which is not enough to estimate covariances and "
                "continue the assimilation."
            )

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
        self.d_obs_uc = (
            self.obs.reshape(-1, 1)
            + np.sqrt(inflation_factor)
            * self.cov_obs.sample_mvnormal((self.n_ensemble,), self.rng).T
        )

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
                self.inversion_type,
                inflation_factor,
                self.cov_obs,
                self.d_obs_uc,
                self.d_pred,
                self.m_prior,
                C_DD_localization=self.C_DD_localization,
                C_MD_localization=self.C_MD_localization,
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
        worker = partial(
            _run_batch_update,
            inflation_factor=inflation_factor,
            batch_size=self.batch_size,
            m_dim=self.m_dim,
            m_prior=self.m_prior,
            inversion_type=self.inversion_type,
            cov_obs=self.cov_obs,
            d_obs_uc=self.d_obs_uc,
            d_pred=self.d_pred,
            C_DD_localization=self.C_DD_localization,
            C_MD_localization=self.C_MD_localization,
        )
        if self.is_parallel_analyse_step:
            results = Parallel(n_jobs=-1)(
                delayed(worker)(index) for index in range(self.n_batches)
            )
        else:
            results = [worker(index) for index in range(self.n_batches)]

        for index, res in enumerate(results):
            _slice = slice(
                index * self.batch_size,
                min((index + 1) * self.batch_size, self.m_dim),
            )
            m_pred[_slice, :] = res
        return m_pred

    def _apply_bounds(self, m_pred: NDArrayFloat) -> NDArrayFloat:
        """Apply bounds constraints to the adjusted parameters."""
        return np.clip(m_pred.T, self.m_bounds[:, 0], self.m_bounds[:, 1]).T
