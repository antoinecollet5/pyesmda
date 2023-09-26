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
# Every inversion function has the form
# inversion_<exact/approximate>_<name>
from __future__ import annotations

from enum import Enum
from typing import List, Optional

import numpy as np
import scipy as sp  # type: ignore
from scipy.sparse import spmatrix  # type: ignore

from pyesmda.utils import (
    NDArrayFloat,
    empirical_cross_covariance,
    get_anomaly_matrix,
    np_cache,
)


class ESMDAInversionType(str, Enum):
    """
    Inversion type for the computation of
    $C_{md} (C_{dd} + \alpha * C_{d})^{-1} (d - Y)$.

    It is a hashable string enum and can be iterated.

    Available inversion types are:
    - naive: direct inversion of C_DD + alpha * C_D
    - exact_cholesky
    - exact_lstq
    - exact_rescaled
    - exact_subspace
    - subspace: sub
    - subspace_rescaled: Same as subspace but with a rescaling procedure to avoid loss
    of information during truncation of small singular values
    (see :cite:t:`evensenSamplingStrategiesSquare2004`).
    """

    # direct inversion (CD)
    NAIVE = "naive"
    # only if cdd is diagonal
    EXACT_CHOLESKY = "exact_cholesky"
    # for big data assimilation this is the recommended method
    EXACT_LSTSQ = "exact_lstq"
    # for big data assimilation this is the recommended method
    EXACT_RESCALED = "exact_rescaled"
    EXACT_SUBSPACE = "exact_subspace"  # for ...
    SUBSPACE = "subspace"
    SUBSPACE_RESCALED = "subspace_rescaled"  # using full Cdd

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


def get_localized_cdd(Y: NDArrayFloat, dd_corr_mat: Optional[spmatrix]) -> NDArrayFloat:
    """
    Get the empirical auto-correlation matrix $C_{dd}$.

    If provided, the matrix is masked with the provided localization matrix.

    Parameters
    ----------
    m_pred : npt.NDArray[np.float64]
        Ensemble of predicted values with dimensions
        (:math:`N_{obs}`, :math:`N_{e}`).

    """
    C_DD = empirical_cross_covariance(Y, Y)
    if dd_corr_mat is not None:
        return dd_corr_mat.multiply(C_DD)  # type: ignore
    return C_DD


def get_localized_cmd_multi_dot(
    X: NDArrayFloat,
    Y: NDArrayFloat,
    *args: NDArrayFloat,
    md_corr_mat: Optional[spmatrix] = None,
) -> NDArrayFloat:
    """_summary_

    Parameters
    ----------
    X : NDArrayFloat
        _description_
    Y : NDArrayFloat
        _description_
    md_corr_mat : Optional[spmatrix], optional
        _description_, by default None

    Returns
    -------
    NDArrayFloat
        _description_
    """
    X_shift = get_anomaly_matrix(X)
    Y_shift = get_anomaly_matrix(Y)

    if md_corr_mat is not None:
        return np.linalg.multi_dot(  # type: ignore
            [
                md_corr_mat.multiply(  # type: ignore
                    X_shift.dot(Y_shift.T)  # type: ignore
                ).toarray(),
                *args,
            ]
        )
    return np.linalg.multi_dot([X_shift, Y_shift.T, *args])  # type: ignore


@np_cache()
def get_inv(input: NDArrayFloat) -> NDArrayFloat:
    """Get the inversed matrix."""
    return np.linalg.inv(input)


def inversion(
    invertion_type: ESMDAInversionType,
    inflation_factor: float,
    cov_obs: NDArrayFloat,
    cov_obs_cholesky: NDArrayFloat,
    obs_uc: NDArrayFloat,
    d_pred: NDArrayFloat,
    s_ens: NDArrayFloat,
    dd_corr_mat: Optional[spmatrix] = None,
    md_corr_mat: Optional[spmatrix] = None,
    truncation: float = 0.99,
) -> NDArrayFloat:
    r"""
    Computes C_MD @ inv(C_DD + alpha * C_D) @ (D - Y).

    Parameters
    ----------
    invertion_type : ESMDAInversionType
        Type of inversion. See :class:`ESMDAInversionType` for available methods.
    inflation_factor : float
        Inflation factor :math:`\alpha` for `cov_obs`, the covariance matrix of
        observed data measurement errors.
    cov_obs : NDArrayFloat
        Covariance matrix of observed data measurement errors with dimensions
        (:math:`N_{obs}`, :math:`N_{obs}`). Also denoted :math:`R`.
    cov_obs_cholesky : NDArrayFloat
        Cholesky factorization (upper) of :variable:`cov_obs`.
    obs_uc : NDArrayFloat
        Matrix of perturbed observations with shape (:math:`N_{obs}`, :math:`N_{e}`).
    d_pred : NDArrayFloat
        Ensemble of predicted values with shape (:math:`N_{obs}`, :math:`N_{e}`).
    m_pred : npt.NDArray[np.float64]
        Ensemble of adjusted parameters with dimensions
        (:math:`N_{m}`, :math:`N_{e}`).
    dd_corr_mat: Optional[spmatrix]
        Correlation matrix based on spatial and temporal distances between
        observations and observations :math:`\rho_{DD}`. It is used to localize the
        autocovariance matrix of predicted data by applying an elementwise
        multiplication by this matrix.
        Expected dimensions are (:math:`N_{obs}`, :math:`N_{obs}`). The default is None.
    md_corr_mat: Optional[spmatrix]
        Correlation matrix based on spatial and temporal distances between
        parameters and observations :math:`\rho_{MD}`. It is used to localize the
        cross-covariance matrix between the forecast state vector (parameters)
        and predicted data by applying an elementwise
        multiplication by this matrix.
        Expected dimensions are (:math:`N_{m}`, :math:`N_{obs}`). The default is None.
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
    """
    return {  # type: ignore
        ESMDAInversionType.NAIVE: inversion_exact_naive,
        ESMDAInversionType.EXACT_CHOLESKY: inversion_exact_cholesky,
        ESMDAInversionType.EXACT_LSTSQ: inversion_exact_lstsq,
        ESMDAInversionType.EXACT_RESCALED: inversion_exact_rescaled,
        ESMDAInversionType.EXACT_SUBSPACE: inversion_exact_subspace_woodbury,
        ESMDAInversionType.SUBSPACE_RESCALED: inversion_rescaled_subspace,
        ESMDAInversionType.SUBSPACE: inversion_subspace,
    }[invertion_type](
        inflation_factor=inflation_factor,
        C_D=cov_obs,
        C_D_L=cov_obs_cholesky,
        D=obs_uc,
        Y=d_pred,
        X=s_ens,
        dd_corr_mat=dd_corr_mat,
        md_corr_mat=md_corr_mat,
        truncation=truncation,
    )


def inversion_exact_naive(
    *,
    inflation_factor: float,
    C_D: NDArrayFloat,
    D: NDArrayFloat,
    Y: NDArrayFloat,
    X: NDArrayFloat,
    dd_corr_mat: Optional[spmatrix] = None,
    md_corr_mat: Optional[spmatrix] = None,
    **kwargs,
) -> NDArrayFloat:
    """Naive inversion, used for testing only.

    Computes C_MD @ inv(C_DD + inflation_factor * C_D) @ (D - Y) naively.
    """
    # Naive implementation of Equation (3) in Emerick (2013)
    C_MD = empirical_cross_covariance(X, Y)

    if md_corr_mat is not None:
        C_MD = md_corr_mat.multiply(C_MD)  # type: ignore
    C_DD = get_localized_cdd(Y, dd_corr_mat)
    C_D = np.diag(C_D) if C_D.ndim == 1 else C_D

    return C_MD @ sp.linalg.inv(C_DD + inflation_factor * C_D) @ (D - Y)  # type: ignore


def inversion_exact_cholesky(
    *,
    inflation_factor: float,
    C_D: NDArrayFloat,
    D: NDArrayFloat,
    Y: NDArrayFloat,
    X: NDArrayFloat,
    dd_corr_mat: Optional[spmatrix] = None,
    md_corr_mat: Optional[spmatrix] = None,
    **kwargs,
) -> NDArrayFloat:
    """Computes an exact inversion using `sp.linalg.solve`, which uses a
    Cholesky factorization in the case of symmetric, positive definite matrices.

    The goal is to compute: C_MD @ inv(C_DD + \alpha * C_D) @ (D - Y)

    First we solve (C_DD + \alpha * C_D) @ K = (D - Y) for K, so that
    K = inv(C_DD + alpha * C_D) @ (D - Y), then we compute
    C_MD @ K, but we don't explicitly form C_MD, since it might be more
    efficient to perform the matrix products in another order.
    """
    C_DD = get_localized_cdd(Y, dd_corr_mat)

    # Arguments for sp.linalg.solve
    solver_kwargs = {
        "overwrite_a": True,
        "overwrite_b": True,
        "assume_a": "pos",  # Assume positive definite matrix (use cholesky)
        "lower": False,  # Only use the upper part while solving
    }

    # Compute K := sp.linalg.inv(C_DD + alpha * C_D) @ (D - Y)
    if C_D.ndim == 2:
        # C_D is a covariance matrix
        C_DD += inflation_factor * C_D  # Save memory by mutating
        K: NDArrayFloat = sp.linalg.solve(C_DD, (D - Y), **solver_kwargs)
    else:
        # C_D is an array, so add it to the diagonal without forming diag(C_D)
        C_DD.flat[:: C_DD.shape[1] + 1] += inflation_factor * C_D
        K = sp.linalg.solve(C_DD, (D - Y), **solver_kwargs)

    return get_localized_cmd_multi_dot(X, Y, K, md_corr_mat=md_corr_mat)


def inversion_exact_lstsq(
    *,
    inflation_factor: float,
    C_D: NDArrayFloat,
    D: NDArrayFloat,
    Y: NDArrayFloat,
    X: NDArrayFloat,
    dd_corr_mat: Optional[spmatrix] = None,
    md_corr_mat: Optional[spmatrix] = None,
    **kwargs,
) -> NDArrayFloat:
    """Computes inversion using least squares. While this method can deal with
    rank-deficient C_D, it should not be used since it's very slow.
    """
    C_DD = get_localized_cdd(Y, dd_corr_mat)

    # A covariance matrix was given
    if C_D.ndim == 2:
        lhs = C_DD + inflation_factor * C_D
    # A diagonal covariance matrix was given as a vector
    else:
        lhs = C_DD
        lhs.flat[:: lhs.shape[0] + 1] += inflation_factor * C_D

    # K = lhs^-1 @ (D - Y)
    # lhs @ K = (D - Y)
    K, *_ = sp.linalg.lstsq(
        lhs, D - Y, overwrite_a=True, overwrite_b=True, lapack_driver="gelsy"
    )
    return get_localized_cmd_multi_dot(X, Y, K, md_corr_mat=md_corr_mat)


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
    assert np.all(
        np.diff(singular_values) <= 0
    ), "Singular values must be sorted decreasing"
    assert 0 < truncation <= 1, "Threshold must be in range (0, 1]"
    singular_values = np.array(singular_values, dtype=float)

    # Take cumulative sum and normalize
    cumsum = np.cumsum(singular_values)
    cumsum /= cumsum[-1]
    return int(np.searchsorted(cumsum, v=truncation, side="left") + 1)


def inversion_exact_rescaled(
    *,
    inflation_factor: float,
    C_D: NDArrayFloat,
    C_D_L: NDArrayFloat,
    D: NDArrayFloat,
    Y: NDArrayFloat,
    X: NDArrayFloat,
    dd_corr_mat: Optional[spmatrix] = None,
    md_corr_mat: Optional[spmatrix] = None,
    truncation: float = 0.99,
    **kwargs,
) -> NDArrayFloat:
    """Compute a rescaled inversion.

    See Appendix A.1 in :cite:t:`emerickHistoryMatchingTimelapse2012`
    for details regarding this approach.
    """
    C_DD = get_localized_cdd(Y, dd_corr_mat)

    if C_D.ndim == 2:
        C_D_L_inv, _ = sp.linalg.lapack.dtrtri(
            C_D_L, lower=0, overwrite_c=0
        )  # Invert lower triangular using BLAS routine
        C_D_L_inv /= np.sqrt(inflation_factor)

        # Eqn (59). Form C_tilde
        # TODO: Use BLAS routine for triangular times dense matrix
        C_tilde = sp.linalg.blas.strmm(alpha=1, a=C_D_L_inv, b=C_DD, lower=0)
        C_tilde = C_D_L_inv @ C_DD @ C_D_L_inv.T
        C_tilde.flat[:: C_tilde.shape[0] + 1] += 1.0  # Add to diagonal

    # When C_D is a diagonal covariance matrix, there is no need to perform
    # the cholesky factorization
    else:
        C_D_L_inv = 1 / np.sqrt(C_D * inflation_factor)
        C_tilde = (C_D_L_inv * (C_DD * C_D_L_inv).T).T
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
    term = C_D_L_inv.T @ U_r if C_D.ndim == 2 else (C_D_L_inv * U_r.T).T

    return get_localized_cmd_multi_dot(
        X, Y, term / s_r, term.T, (D - Y), md_corr_mat=md_corr_mat
    )


def inversion_exact_subspace_woodbury(
    *,
    inflation_factor: float,
    C_D: NDArrayFloat,
    D: NDArrayFloat,
    Y: NDArrayFloat,
    X: NDArrayFloat,
    dd_corr_mat: Optional[spmatrix] = None,
    md_corr_mat: Optional[spmatrix] = None,
    **kwargs,
) -> NDArrayFloat:
    """Use the Woodbury lemma to compute the inversion.

    This approach uses the Woodbury lemma to compute:
        C_MD @ inv(C_DD + inflation_factor * C_D) @ (D - Y)

    Since C_DD = U @ U.T, where U := center(Y) / sqrt(N_e - 1), we can use:

    V = inflation_factor * C_D

    (V + U @ U.T)^-1 = V^-1 - V^-1 @ U @ (1 + U.T @ V^-1 @ U )^-1 @ U.T @ V^-1

    to compute inv(C_DD + inflation_factor * C_D).
    """
    # TODO: If regularization -> we can try to apply the localization
    # and then to use cholesky afterwards ?
    Y_shift = get_anomaly_matrix(Y)

    # A full covariance matrix was given
    if C_D.ndim == 2:
        # Invert C_D -> we use a cached function
        C_D_inv = get_inv(C_D) / inflation_factor

        # Compute the center part of the rhs in woodburry
        center = np.linalg.multi_dot([Y_shift.T, C_D_inv, Y_shift])
        center.flat[:: center.shape[0] + 1] += 1.0  # Add to diagonal

        # Compute the symmetric term of the rhs in woodbury
        term = C_D_inv @ Y_shift

        # Compute the woodbury inversion, then return
        inverted = C_D_inv - np.linalg.multi_dot([term, sp.linalg.inv(center), term.T])

        return get_localized_cmd_multi_dot(
            X, Y, inverted, (D - Y), md_corr_mat=md_corr_mat
        )

    # A diagonal covariance matrix was given as a 1D array.
    # Same computation as above, but exploit the diagonal structure
    else:
        C_D_inv = 1 / (C_D * inflation_factor)  # Invert diagonal
        center = np.linalg.multi_dot([Y_shift.T * C_D_inv, Y_shift])
        center.flat[:: center.shape[0] + 1] += 1.0
        UT_D = Y_shift.T * C_D_inv
        inverted = np.diag(C_D_inv) - np.linalg.multi_dot(
            [UT_D.T, sp.linalg.inv(center), UT_D]
        )
        return get_localized_cmd_multi_dot(
            X, Y, inverted, (D - Y), md_corr_mat=md_corr_mat
        )


def inversion_subspace(
    *,
    inflation_factor: float,
    C_D: NDArrayFloat,
    D: NDArrayFloat,
    Y: NDArrayFloat,
    X: NDArrayFloat,
    dd_corr_mat: Optional[spmatrix] = None,
    md_corr_mat: Optional[spmatrix] = None,
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

    """
    # TODO: localization

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
    if C_D.ndim == 1:
        X1 = (N_e - 1) * np.linalg.multi_dot(
            [U_r_w_inv.T * C_D * inflation_factor, U_r_w_inv]
        )
    else:
        X1 = (N_e - 1) * np.linalg.multi_dot(
            [U_r_w_inv.T, inflation_factor * C_D, U_r_w_inv]
        )

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
    return get_localized_cmd_multi_dot(
        X * (N_e - 1), Y, (term / (1 + T)), term.T, (D - Y), md_corr_mat=md_corr_mat
    )


def inversion_rescaled_subspace(
    *,
    inflation_factor: float,
    C_D: NDArrayFloat,
    C_D_L: NDArrayFloat,
    D: NDArrayFloat,
    Y: NDArrayFloat,
    X: NDArrayFloat,
    dd_corr_mat: Optional[spmatrix] = None,
    md_corr_mat: Optional[spmatrix] = None,
    truncation: float = 0.99,
    **kwargs,
) -> NDArrayFloat:
    """
    See Appendix A.2 in :cite:t:`emerickHistoryMatchingTimelapse2012`.

    Subspace inversion with rescaling.
    """
    N_n, N_e = Y.shape
    Y_shift = Y - np.mean(Y, axis=1, keepdims=True)  # Subtract average

    if C_D.ndim == 2:
        # Here C_D_L is C^{1/2} in equation (57)
        # assert np.allclose(C_D_L @ C_D_L.T, C_D * alpha)
        print(inflation_factor)
        C_D_L_inv, _ = sp.linalg.lapack.dtrtri(
            C_D_L * np.sqrt(inflation_factor), lower=0, overwrite_c=0
        )  # Invert upper triangular

        # Use BLAS to compute product of upper triangular matrix C_D_L_inv and Y_shift
        # This line is equal to C_D_L_inv @ Y_shift
        C_D_L_times_Y_shift = sp.linalg.blas.dtrmm(
            alpha=1.0, a=C_D_L_inv, b=Y_shift, lower=0
        )

        C_D_L_times_Y_shift = C_D_L_inv @ Y_shift

    else:
        # Same as above, but C_D is a vector
        C_D_L_inv = 1 / np.sqrt(
            inflation_factor * C_D
        )  # Invert the Cholesky factor a diagonal
        C_D_L_times_Y_shift = (Y_shift.T * C_D_L_inv).T

    U, w, _ = sp.linalg.svd(C_D_L_times_Y_shift, overwrite_a=True, full_matrices=False)
    idx = singular_values_to_keep(w, truncation=truncation)

    # assert np.allclose(VT @ VT.T, np.eye(VT.shape[0]))
    N_r = min(N_n, N_e - 1, idx)  # Number of values in SVD to keep
    U_r, w_r = U[:, :N_r], w[:N_r]

    # Eqn (78) - taking into account that C_D_L_inv could be an array
    term = C_D_L_inv.T @ (U_r / w_r) if C_D.ndim == 2 else ((U_r / w_r).T * C_D_L_inv).T
    T_r = (N_e - 1) / w_r**2  # Equation (79)
    diag = 1 / (1 + T_r)

    # Note: need to multiply X by (N_e - 1) to compensate for the anomaly matrix
    # computation
    return get_localized_cmd_multi_dot(
        X * (N_e - 1),
        Y,
        (term * diag),
        term.T,
        (D - Y),
        md_corr_mat=md_corr_mat,
    )
