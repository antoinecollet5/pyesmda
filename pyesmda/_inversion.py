# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021-2026 Antoine COLLET

"""
ESMDA inversion functions.

Note: most of this code has been copied from the implementation of
iterative_ensemble_smoother and as been writtent by ...
See @

All of these functions compute (exactly, or approximately), the product

 C_MD @ inv(C_DD + inflation_factor * C_D) @ (D - Y)

where C_MD = empirical_cross_covariance(X, Y) = center(X) @ center(Y).T
              / (X.shape[1] - 1)
      C_DD = empirical_cross_covariance(Y, Y) = center(Y) @ center(Y).T
              / (Y.shape[1] - 1)

The methods can be classified as
  - exact : with truncation=1.0, these methods compute the exact solution
  - exact : with truncation<1.0, these methods may approximate the solution
  - approximate: if ensemble_members <= num_outputs, then the solution is
                 always approximated, regardless of the truncation
  - approximate: if ensemble_members > num_outputs, then the solution is
                 exact when truncation is 1.0

"""

from __future__ import annotations

import functools
import warnings
from enum import Enum
from typing import Callable, Dict, List

import covmats
import numpy as np
import scipy as sp

from pyesmda._localization import LocalizationStrategy, NoLocalization
from pyesmda._utils import (
    NDArrayFloat,
    get_anomaly_matrix,
)


@functools.cache
def cholesky_upper(cov: covmats.CovarianceMatrix) -> NDArrayFloat:
    """
    Return the upper Cholesky factor ``U`` of the dense covariance matrix.

    ``U`` is upper-triangular and satisfies ``U.T @ U == cov.todense()``.

    """
    return sp.linalg.cholesky(cov.todense(), lower=False)


class ESMDAInversionType(str, Enum):
    r"""
    Inversion type for the computation of
    :math:`\mathbf{C}_{\mathrm{md}} (\mathbf{C}_{\mathrm{dd}} +
    \alpha \mathbf{C}_{\mathrm{d}})^{-1} (\mathbf{d} - \mathbf{Y})`.

    It is a hashable string enum and can be iterated.

    Available inversion types are:
        - **naive**: direct inversion of C_DD + alpha * C_D
        - **cholesky**: perform the cholesky factorization of C_DD + alpha * C_D
        - **lstsq**: Computes inversion using least squares. While this method can
          deal with rank-deficient C_D, it should not be used since it's very slow
        - **woodbury**: Rely on woodbury lemma to reformulate the problem
        - **rescaled**: rely on truncated singular value decomposition TSVD of C_DD
        - **subspace**: rely on TSVD of U with C_DD = UU^{T}
        - **subspace_rescaled**: Same as subspace but with a rescaling procedure to
          avoid loss of information during truncation of small singular values
          (see :cite:t:`evensenSamplingStrategiesSquare2004`)
    """

    NAIVE = "naive"
    CHOLESKY = "cholesky"
    LSTSQ = "lstsq"
    WOODBURY = "woodbury"
    RESCALED = "rescaled"
    SUBSPACE = "subspace"
    SUBSPACE_RESCALED = "subspace_rescaled"

    def __str__(self) -> str:
        """Return instance value."""
        return self.value

    def __hash__(self) -> int:
        """Return the hash of the value."""
        return hash(self.value)

    def __eq__(self, other: object) -> bool:
        """Return if two instances are equal."""
        if not isinstance(other, ESMDAInversionType) and not isinstance(other, str):
            return False
        return self.value == other

    @classmethod
    def to_list(cls) -> List[ESMDAInversionType]:
        """Return all enums as a list."""
        return list(cls)


# Inversion types for which C_DD_localization is accepted but not currently
# applied. See the warning raised for these in `inversion()` below, and the
# per-function docstrings for why (in short: the Woodbury and subspace
# formulations never materialize the dense C_DD matrix that a
# LocalizationStrategy.localize() call would need to mask).
_INVERSION_TYPES_IGNORING_C_DD_LOCALIZATION = frozenset(
    {
        ESMDAInversionType.WOODBURY,
        ESMDAInversionType.SUBSPACE,
        ESMDAInversionType.SUBSPACE_RESCALED,
    }
)


def inversion(
    inversion_type: ESMDAInversionType,
    inflation_factor: float,
    cov_obs: covmats.CovarianceMatrix,
    obs_uc: NDArrayFloat,
    d_pred: NDArrayFloat,
    s_ens: NDArrayFloat,
    C_DD_localization: LocalizationStrategy = NoLocalization(),
    C_MD_localization: LocalizationStrategy = NoLocalization(),
    truncation: float = 0.99,
    batch_slice: slice = slice(None),
) -> NDArrayFloat:
    r"""
    Computes C_MD @ inv(C_DD + alpha * C_D) @ (D - Y).

    Parameters
    ----------
    inversion_type : ESMDAInversionType
        Type of inversion. See :py:class:`ESMDAInversionType` for available methods.
    inflation_factor : float
        Inflation factor :math:`\alpha` for `cov_obs`, the covariance matrix of
        observed data measurement errors.
    cov_obs : covmats.CovarianceMatrix
        Covariance matrix of observed data measurement errors with dimensions
        (:math:`N_{\mathrm{obs}}`, :math:`N_{\mathrm{obs}}`). Also denoted :math:`R`.
    obs_uc : NDArrayFloat
        Matrix of perturbed observations with shape (:math:`N_{\mathrm{obs}}`,
        :math:`N_{e}`).
    d_pred : NDArrayFloat
        Ensemble of predicted values with shape (:math:`N_{\mathrm{obs}}`,
        :math:`N_{e}`).
    s_ens : NDArrayFloat
        Ensemble of adjusted parameters with dimensions
        (:math:`N_{m}`, :math:`N_{e}`).
    C_DD_localization: LocalizationStrategy
        Localization operator :math:`\rho_{DD}` applied to the predictions
        empirical auto-covariance matrices. Expected dimensions of the operator are
        (:math:`N_{\mathrm{obs}}`, :math:`N_{\mathrm{obs}}`). It can be fixed (defined
        correlation matrix used for all iterations) or adaptive and even user defined.
        See implementations of :py:class:`LocalizationStrategy`.

        Note
        ----
        Not every `inversion_type` currently applies this. See
        :py:data:`_INVERSION_TYPES_IGNORING_C_DD_LOCALIZATION`; a
        :py:class:`UserWarning` is raised when a non-default localization is
        passed for one of those types, since it is silently ignored.
    C_MD_localization : LocalizationStrategy
        Localization operator :math:`\rho_{MD}` applied to the parameters-predictions
        empirical cross-covariance matrices. Expected dimensions of the operator are
        (:math:`N_{m}`, :math:`N_{\mathrm{obs}}`). It can be fixed (defined correlation
        matrix used for all iterations) or adaptive and even user defined.
        See implementations of :py:class:`LocalizationStrategy`. Applied by every
        `inversion_type`.
    truncation : float, optional
        truncation: float
        A value in the range ]0, 1], used to determine the number of
        significant singular values kept when using svd for the inversion
        of $(C_{dd} + \alpha C_{d})$: Only the largest singular values are kept,
        corresponding to this fraction of the sum of the nonzero singular values.
        The goal of truncation is to deal with smaller matrices (dimensionality
        reduction), easier to inverse. The default is 0.99.

    Returns
    -------
    NDArrayFloat
        The update :math:`\delta X`.

    Raises
    ------
    KeyError
        If `inversion_type` is not one of the values in
        :py:class:`ESMDAInversionType`.
    """
    if inversion_type in _INVERSION_TYPES_IGNORING_C_DD_LOCALIZATION and not isinstance(
        C_DD_localization, NoLocalization
    ):
        warnings.warn(
            f"C_DD_localization is not supported by inversion_type="
            f"'{inversion_type}' and will be silently ignored. Use "
            f"inversion_type='naive', 'cholesky', 'lstsq', or 'rescaled' if "
            f"C_DD localization is required.",
            UserWarning,
            stacklevel=2,
        )

    return _INVERSION_DISPATCH[inversion_type](
        inflation_factor=inflation_factor,
        C_D=cov_obs,
        D=obs_uc,
        Y=d_pred,
        X=s_ens,
        C_DD_localization=C_DD_localization,
        C_MD_localization=C_MD_localization,
        truncation=truncation,
        batch_slice=batch_slice,
    )


def _run_batch_update(
    index: int,
    inflation_factor: float,
    batch_size: int,
    m_dim: int,
    m_prior: NDArrayFloat,
    inversion_type: ESMDAInversionType,
    cov_obs: covmats.CovarianceMatrix,
    d_obs_uc: NDArrayFloat,
    d_pred: NDArrayFloat,
    C_DD_localization: LocalizationStrategy,
    C_MD_localization: LocalizationStrategy,
) -> NDArrayFloat:
    """
    Compute the updated parameters for a single batch of rows of `m_prior`.

    This is the function dispatched to worker processes (via
    ``functools.partial``) by ``ESMDABase._local_analyse`` when
    ``is_parallel_analyse_step`` is True. It is also called directly,
    sequentially, when running without parallelism. See
    :py:meth:`pyesmda._base.ESMDABase._local_analyse`.

    Parameters
    ----------
    index : int
        Index of the batch to process, in ``range(n_batches)``.
    inflation_factor : float
        Inflation factor :math:`\\alpha` for `cov_obs` at the current
        assimilation step.
    batch_size : int
        Number of parameters (rows of `m_prior`) processed per batch.
    m_dim : int
        Total number of parameters (rows of `m_prior`), used to clip the
        last batch to the correct size.
    m_prior : NDArrayFloat
        Full, unsliced ensemble of prior parameters, with dimensions
        (:math:`N_{m}`, :math:`N_{e}`). Only the rows selected by this
        batch's slice are updated and returned.
    inversion_type : ESMDAInversionType
        Type of inversion. See :py:class:`ESMDAInversionType`.
    cov_obs : covmats.CovarianceMatrix
        Full, unsliced covariance matrix of observation errors -- identical
        across every batch of a given assimilation step.
    d_obs_uc : NDArrayFloat
        Matrix of perturbed observations, with dimensions
        (:math:`N_{\\mathrm{obs}}`, :math:`N_{e}`).
    d_pred : NDArrayFloat
        Ensemble of predicted values, with dimensions
        (:math:`N_{\\mathrm{obs}}`, :math:`N_{e}`).
    C_DD_localization : LocalizationStrategy
        Localization strategy for the predictions auto-covariance.
    C_MD_localization : LocalizationStrategy
        Localization strategy for the parameters-predictions cross-covariance.

    Returns
    -------
    NDArrayFloat
        Updated parameters for this batch's rows only, with dimensions
        (batch rows, :math:`N_{e}`).
    """
    _slice = slice(index * batch_size, min((index + 1) * batch_size, m_dim))
    return m_prior[_slice, :] + (
        inversion(
            inversion_type,
            inflation_factor,
            cov_obs,
            d_obs_uc,
            d_pred,
            m_prior[_slice, :].reshape(-1, m_prior.shape[-1]),
            C_DD_localization=C_DD_localization,
            C_MD_localization=C_MD_localization,
            truncation=1.0,
            batch_slice=_slice,
        )
    )


def inversion_naive(
    *,
    inflation_factor: float,
    C_D: covmats.CovarianceMatrix,
    D: NDArrayFloat,
    Y: NDArrayFloat,
    X: NDArrayFloat,
    C_DD_localization: LocalizationStrategy = NoLocalization(),
    C_MD_localization: LocalizationStrategy = NoLocalization(),
    batch_slice: slice = slice(None),
    **kwargs,
) -> NDArrayFloat:
    """Naive inversion, used for testing only.

    Computes C_MD @ inv(C_DD + inflation_factor * C_D) @ (D - Y) naively.
    """
    # Naive implementation of Equation (3) in Emerick (2013)
    C_MD = C_MD_localization.localize(X, Y, batch_slice=batch_slice)
    C_DD = C_DD_localization.localize(Y, Y)
    return C_MD @ sp.linalg.inv(C_DD + inflation_factor * C_D.todense()) @ (D - Y)


def inversion_cholesky(
    *,
    inflation_factor: float,
    C_D: covmats.CovarianceMatrix,
    D: NDArrayFloat,
    Y: NDArrayFloat,
    X: NDArrayFloat,
    C_DD_localization: LocalizationStrategy = NoLocalization(),
    C_MD_localization: LocalizationStrategy = NoLocalization(),
    batch_slice: slice = slice(None),
    **kwargs,
) -> NDArrayFloat:
    r"""Computes an exact inversion using `sp.linalg.solve`, which uses a
    Cholesky factorization in the case of symmetric, positive definite matrices.

    The goal is to compute: C_MD @ inv(C_DD + \alpha * C_D) @ (D - Y)

    First we solve (C_DD + \alpha * C_D) @ K = (D - Y) for K, so that
    K = inv(C_DD + alpha * C_D) @ (D - Y), then we compute
    C_MD @ K, but we don't explicitly form C_MD, since it might be more
    efficient to perform the matrix products in another order.
    """
    C_DD = C_DD_localization.localize(Y, Y)

    # Arguments for sp.linalg.solve
    solver_kwargs = {
        "overwrite_a": True,
        "overwrite_b": True,
        "assume_a": "pos",  # Assume positive definite matrix (use cholesky)
        "lower": False,  # Only use the upper part while solving
    }

    # Compute K := sp.linalg.inv(C_DD + alpha * C_D) @ (D - Y)
    if not isinstance(C_D, covmats.CovViaDiagonal):
        # C_D is a covariance matrix
        C_DD += inflation_factor * C_D.todense()  # Save memory by mutating
        K: NDArrayFloat = sp.linalg.solve(C_DD, (D - Y), **solver_kwargs)
    # A diagonal covariance matrix was given as a vector
    else:
        # C_D is an array, so add it to the diagonal without forming diag(C_D)
        C_DD.flat[:: C_DD.shape[1] + 1] += inflation_factor * C_D.get_diagonal()
        K = sp.linalg.solve(C_DD, (D - Y), **solver_kwargs)

    return C_MD_localization.localize_multi_dot(X, Y, K, batch_slice=batch_slice)


def inversion_lstsq(
    *,
    inflation_factor: float,
    C_D: covmats.CovarianceMatrix,
    D: NDArrayFloat,
    Y: NDArrayFloat,
    X: NDArrayFloat,
    C_DD_localization: LocalizationStrategy = NoLocalization(),
    C_MD_localization: LocalizationStrategy = NoLocalization(),
    batch_slice: slice = slice(None),
    **kwargs,
) -> NDArrayFloat:
    """Computes inversion using least squares. While this method can deal with
    rank-deficient C_D, it should not be used since it's very slow.
    """
    C_DD = C_DD_localization.localize(Y, Y)

    # A covariance matrix was given
    if not isinstance(C_D, covmats.CovViaDiagonal):
        C_DD += inflation_factor * C_D.todense()  # Save memory by mutating
    # A diagonal covariance matrix was given as a vector
    else:
        C_DD.flat[:: C_DD.shape[0] + 1] += inflation_factor * C_D.get_diagonal()

    # K = lhs^-1 @ (D - Y)
    # lhs @ K = (D - Y)
    K, *_ = sp.linalg.lstsq(
        C_DD, D - Y, overwrite_a=True, overwrite_b=True, lapack_driver="gelsy"
    )
    return C_MD_localization.localize_multi_dot(X, Y, K, batch_slice=batch_slice)


def inversion_woodbury(
    *,
    inflation_factor: float,
    C_D: covmats.CovarianceMatrix,
    D: NDArrayFloat,
    Y: NDArrayFloat,
    X: NDArrayFloat,
    C_DD_localization: LocalizationStrategy = NoLocalization(),
    C_MD_localization: LocalizationStrategy = NoLocalization(),
    batch_slice: slice = slice(None),
    **kwargs,
) -> NDArrayFloat:
    """Use the Woodbury lemma to compute the inversion.

    This approach uses the Woodbury lemma to compute:
        C_MD @ inv(C_DD + inflation_factor * C_D) @ (D - Y)

    Since C_DD = U @ U.T, where U := center(Y) / sqrt(N_e - 1), we can use:

    V = inflation_factor * C_D

    (V + U @ U.T)^-1 = V^-1 - V^-1 @ U @ (1 + U.T @ V^-1 @ U )^-1 @ U.T @ V^-1

    to compute inv(C_DD + inflation_factor * C_D).

    Note
    ----
    `C_DD_localization` is NOT applied here: C_DD is never materialized as a
    dense matrix (it stays implicit via the low-rank U @ U.T factors used
    above), so there is no dense matrix for
    `LocalizationStrategy.localize()` to mask elementwise. Only
    `C_MD_localization` is applied. See
    :py:data:`_INVERSION_TYPES_IGNORING_C_DD_LOCALIZATION`.
    """
    # and then to use cholesky afterwards ?
    Y_shift = get_anomaly_matrix(Y)
    rhs = D - Y

    # Symmetric term of the rhs in woodbury.
    term = C_D.solve(Y_shift / inflation_factor)

    # Center part of the rhs in woodbury
    center = Y_shift.T @ term
    center.flat[:: center.shape[0] + 1] += 1.0  # Add to diagonal

    correction = np.linalg.multi_dot([term, sp.linalg.inv(center), term.T, rhs])

    return C_MD_localization.localize_multi_dot(
        X,
        Y,
        C_D.solve(rhs / inflation_factor) - correction,
        batch_slice=batch_slice,
    )


def singular_values_to_keep(
    singular_values: NDArrayFloat, truncation: float = 1.0
) -> int:
    """Find the index of the singular values to keep when truncating.

    Examples
    --------
    >>> singular_values = np.array([3, 2, 1, 0, 0, 0])
    >>> i = singular_values_to_keep(singular_values, truncation=1.0)
    >>> singular_values[:i]
    array([3, 2, 1])

    >>> singular_values = np.array([4, 3, 2, 1])
    >>> i = singular_values_to_keep(singular_values, truncation=1.0)
    >>> singular_values[:i]
    array([4, 3, 2, 1])

    >>> singular_values = np.array([4, 3, 2, 1])
    >>> singular_values_to_keep(singular_values, truncation=0.95)
    4
    >>> singular_values_to_keep(singular_values, truncation=0.9)
    3
    >>> singular_values_to_keep(singular_values, truncation=0.7)
    2

    """
    assert np.all(np.diff(singular_values) <= 0), (
        "Singular values must be sorted decreasing"
    )
    assert 0 < truncation <= 1, "Threshold must be in range (0, 1]"
    singular_values = np.array(singular_values, dtype=float)

    # Take cumulative sum and normalize
    cumsum = np.cumsum(singular_values)
    cumsum /= cumsum[-1]
    return int(np.searchsorted(cumsum, v=truncation, side="left") + 1)


def inversion_rescaled(
    *,
    inflation_factor: float,
    C_D: covmats.CovarianceMatrix,
    D: NDArrayFloat,
    Y: NDArrayFloat,
    X: NDArrayFloat,
    C_DD_localization: LocalizationStrategy = NoLocalization(),
    C_MD_localization: LocalizationStrategy = NoLocalization(),
    batch_slice: slice = slice(None),
    truncation: float = 0.99,
    **kwargs,
) -> NDArrayFloat:
    """Compute a rescaled inversion.

    See Appendix A.1 in :cite:t:`emerickHistoryMatchingTimelapse2012`
    for details regarding this approach.
    """
    C_DD = C_DD_localization.localize(Y, Y)

    # Eqn (59). Form C_tilde
    # C_tilde = C_D.whiten(C_D.whiten(C_DD / inflation_factor))
    # C_tilde.flat[:: C_tilde.shape[0] + 1] += 1.0  # Add to diagonal

    if not isinstance(C_D, covmats.CovViaDiagonal):
        C_D_U = cholesky_upper(C_D)

        C_D_U_inv, _ = sp.linalg.lapack.dtrtri(
            C_D_U, lower=0, overwrite_c=0
        )  # Invert upper triangular using BLAS routine
        C_D_U_inv /= np.sqrt(inflation_factor)

        # Eqn (59). Form C_tilde
        C_tilde = C_D_U_inv.T @ C_DD @ C_D_U_inv
        C_tilde.flat[:: C_tilde.shape[0] + 1] += 1.0  # Add to diagonal

    # When C_D is a diagonal covariance matrix, there is no need to perform
    # the cholesky factorization
    else:
        C_D_U_inv = 1.0 / np.sqrt(C_D.get_diagonal() * inflation_factor)
        C_tilde = (C_D_U_inv * (C_DD * C_D_U_inv).T).T
        C_tilde.flat[:: C_tilde.shape[0] + 1] += 1.0  # Add to diagonal

    # Eqn (60). Compute SVD, which is equivalent to taking eigendecomposition
    # since C_tilde is PSD. Using eigh() is faster than svd().
    # Note that svd() returns eigenvalues in decreasing order, while eigh()
    # returns eigenvalues in increasing order.
    # driver="evr" => fastest option
    s, U = sp.linalg.eigh(C_tilde, driver="evr", overwrite_a=True)
    # Truncate the SVD ( U_r @ np.diag(s_r) @ U_r.T == C_tilde )
    # N_n is the number of observations
    # N_e is the number of members in the ensemble
    N_n, N_e = Y.shape

    idx = singular_values_to_keep(s[::-1], truncation=truncation)
    N_r = min(N_n, N_e - 1, idx)  # Number of values in SVD to keep
    U_r, s_r = U[:, -N_r:], s[-N_r:]
    # U_r @ np.diag(s_r) @ U_r.T == C_tilde

    # Eqn (61). Compute symmetric term once first, then multiply together and
    # finally multiply with (D - Y)
    # term == L^-T @ U_r == (U.T)^-T @ U_r == C_D_U_inv @ U_r (no transpose
    # here -- see the comment above on the C_tilde line for the convention).
    term = (
        C_D_U_inv @ U_r
        if not isinstance(C_D, covmats.CovViaDiagonal)
        else (C_D_U_inv * U_r.T).T
    )
    # term = C_D.whiten(U_r)

    return C_MD_localization.localize_multi_dot(
        X, Y, term / s_r, term.T, (D - Y), batch_slice=batch_slice
    )


def inversion_subspace(
    *,
    inflation_factor: float,
    C_D: covmats.CovarianceMatrix,
    D: NDArrayFloat,
    Y: NDArrayFloat,
    X: NDArrayFloat,
    C_DD_localization: LocalizationStrategy = NoLocalization(),
    C_MD_localization: LocalizationStrategy = NoLocalization(),
    batch_slice: slice = slice(None),
    truncation: float = 0.99,
    **kwargs,
) -> NDArrayFloat:
    """See Appendix A.2 in :cite:t:`emerickHistoryMatchingTimelapse2012`.

    See :cite:t:`evensenSamplingStrategiesSquare2004`.

    This is an approximate solution. The approximation is that when
    U, w, V.T = svd(Y_shift)
    then we assume that U @ U.T = I.
    This is not true in general, for instance:

    >>> Y = np.array([[2, 0],
    ...               [0, 0],
    ...               [0, 0]])
    >>> Y_shift = Y - np.mean(Y, axis=1, keepdims=True)  # Subtract average
    >>> Y_shift
    array([[ 1., -1.],
           [ 0.,  0.],
           [ 0.,  0.]])
    >>> U, w, VT = sp.linalg.svd(Y_shift)
    >>> U, w
    (array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]]), array([1.41421356, 0.        ]))
    >>> U[:, :1] @ np.diag(w[:1]) @ VT[:1, :] # Reconstruct Y_shift
    array([[ 1., -1.],
           [ 0.,  0.],
           [ 0.,  0.]])
    >>> U[:, :1] @ U[:, :1].T # But U_r @ U_r.T != I
    array([[1., 0., 0.],
           [0., 0., 0.],
           [0., 0., 0.]])

    Note
    ----
    `C_DD_localization` is NOT currently applied by this function -- see
    :py:data:`_INVERSION_TYPES_IGNORING_C_DD_LOCALIZATION`. Only
    `C_MD_localization` is applied.
    """
    # N_n is the number of observations
    # N_e is the number of members in the ensemble
    N_n, N_e = Y.shape

    # Subtract the mean of every observation, see Eqn (67)
    Y_shift = Y - np.mean(Y, axis=1, keepdims=True)  # Subtract average

    # Eqn (68)
    # TODO: Approximately 50% of the time in the function is spent here
    # consider using randomized svd for further speed gains
    U, w, _ = sp.linalg.svd(Y_shift, overwrite_a=True, full_matrices=False)

    # Clip the singular value decomposition
    idx = singular_values_to_keep(w, truncation=truncation)
    N_r = min(N_n, N_e - 1, idx)  # Number of values in SVD to keep
    U_r, w_r = U[:, :N_r], w[:N_r]

    # Eqn (70). First compute the symmetric term, then form X
    U_r_w_inv = U_r / w_r
    X1 = (N_e - 1) * inflation_factor * U_r_w_inv.T @ (C_D @ U_r_w_inv)

    # Eqn (72)
    # Z, T, _ = sp.linalg.svd(X, overwrite_a=True, full_matrices=False)
    # Compute SVD, which is equivalent to taking eigendecomposition
    # since X is PSD. Using eigh() is faster than svd().
    # Note that svd() returns eigenvalues in decreasing order, while eigh()
    # returns eigenvalues in increasing order.
    # driver="evr" => fastest option
    T, Z = sp.linalg.eigh(X1, driver="evr", overwrite_a=True)

    # Eqn (74).
    # C^+ = (N_e - 1) hat{C}^+
    #     = (N_e - 1) (U / w @ Z) * (1 / (1 + T)) (U / w @ Z)^T
    #     = (N_e - 1) (term) * (1 / (1 + T)) (term)^T
    # and finally we multiiply by (D - Y)
    term = U_r_w_inv @ Z

    # Note: need to multiply X by (N_e - 1) to compensate for the anomaly matrix
    # computation
    return C_MD_localization.localize_multi_dot(
        X * (N_e - 1), Y, (term / (1 + T)), term.T, (D - Y), batch_slice=batch_slice
    )


def inversion_rescaled_subspace(
    *,
    inflation_factor: float,
    C_D: covmats.CovarianceMatrix,
    D: NDArrayFloat,
    Y: NDArrayFloat,
    X: NDArrayFloat,
    C_DD_localization: LocalizationStrategy = NoLocalization(),
    C_MD_localization: LocalizationStrategy = NoLocalization(),
    batch_slice: slice = slice(None),
    truncation: float = 0.99,
    **kwargs,
) -> NDArrayFloat:
    """
    See Appendix A.2 in :cite:t:`emerickHistoryMatchingTimelapse2012`.

    Subspace inversion with rescaling.

    Note
    ----
    `C_DD_localization` cannot be applied directly by this function -- see
    :py:data:`_INVERSION_TYPE_FALLBACK_FOR_C_DD_LOCALIZATION`. `C_MD_localization`
    is always applied directly. If this function is reached via
    :func:`inversion` with a non-default `C_DD_localization`, that call
    transparently falls back to `inversion_rescaled` instead. Calling
    `inversion_rescaled_subspace` directly (bypassing `inversion()`) does
    not get this fallback and will simply ignore `C_DD_localization`.
    """
    N_n, N_e = Y.shape
    Y_shift = Y - np.mean(Y, axis=1, keepdims=True)  # Subtract average
    if not isinstance(C_D, covmats.CovViaDiagonal):
        C_D_U = cholesky_upper(C_D)
        # get_cholesky() returns the UPPER factor U from
        # scipy.linalg.cholesky(A, lower=False), satisfying U.T @ U == C_D
        # (NOT U @ U.T == C_D). Writing L := U.T gives the usual "lower
        # Cholesky" whitening factor, L @ L.T == C_D * alpha here (alpha
        # folded in via the sqrt(inflation_factor) scaling below), so
        # whitening Y_shift by C_D is L^-1 @ Y_shift
        #   = (U.T)^-1 @ Y_shift == C_D_U_inv.T @ Y_shift
        # (since C_D_U_inv := U^-1, so (U.T)^-1 == C_D_U_inv.T). Verified to
        # machine precision against inversion_naive for non-diagonal C_D.
        C_D_U_inv, _ = sp.linalg.lapack.dtrtri(
            C_D_U * np.sqrt(inflation_factor), lower=0, overwrite_c=0
        )  # Invert upper triangular

        C_D_U_times_Y_shift = C_D_U_inv.T @ Y_shift

    else:
        # Same as above, but C_D is a vector
        C_D_U_inv = 1 / np.sqrt(
            inflation_factor * C_D.get_diagonal()
        )  # Invert the Cholesky factor a diagonal
        C_D_U_times_Y_shift = (Y_shift.T * C_D_U_inv).T

    U, w, _ = sp.linalg.svd(C_D_U_times_Y_shift, overwrite_a=True, full_matrices=False)
    idx = singular_values_to_keep(w, truncation=truncation)

    # assert np.allclose(VT @ VT.T, np.eye(VT.shape[0]))
    N_r = min(N_n, N_e - 1, idx)  # Number of values in SVD to keep
    U_r, w_r = U[:, :N_r], w[:N_r]

    # Eqn (78) - taking into account that C_D_U_inv could be an array
    term = (
        C_D_U_inv @ (U_r / w_r)
        if not isinstance(C_D, covmats.CovViaDiagonal)
        else ((U_r / w_r).T * C_D_U_inv).T
    )
    T_r = (N_e - 1) / w_r**2  # Equation (79)
    diag = 1 / (1 + T_r)

    # Note: need to multiply X by (N_e - 1) to compensate for the anomaly matrix
    # computation
    return C_MD_localization.localize_multi_dot(
        X * (N_e - 1), Y, (term * diag), term.T, (D - Y), batch_slice=batch_slice
    )


# Dispatch table used by `inversion()`. Defined at the end of the module,
# after every inversion_* function, purely for readability (so each
# function can be read top-to-bottom without forward references); this is
# safe because `inversion()` only looks this name up at call time, once the
# whole module has already finished loading, regardless of where in the
# module it's defined.
_INVERSION_DISPATCH: Dict[ESMDAInversionType, Callable[..., NDArrayFloat]] = {
    ESMDAInversionType.NAIVE: inversion_naive,
    ESMDAInversionType.CHOLESKY: inversion_cholesky,
    ESMDAInversionType.LSTSQ: inversion_lstsq,
    ESMDAInversionType.WOODBURY: inversion_woodbury,
    ESMDAInversionType.RESCALED: inversion_rescaled,
    ESMDAInversionType.SUBSPACE: inversion_subspace,
    ESMDAInversionType.SUBSPACE_RESCALED: inversion_rescaled_subspace,
}
