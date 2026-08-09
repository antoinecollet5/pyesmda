# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021-2026 Antoine COLLET

"""
Correlation and localization functions for ensemble smoother methods (ES-MDA).

This module provides:

- Localization strategies (:class:`LocalizationStrategy` and its
  subclasses) used to regularize empirical cross-covariance matrices
  computed from an ensemble, either through a fixed correlation matrix
  or through adaptive, correlation-based schemes.
- Utilities to convert covariance matrices to correlation matrices
  (:func:`cov_to_corr`).
- Correlation transforms (thresholding, tempering) used by adaptive
  localization.
- Tapering/weighting functions based on distance, including the
  Gaspari-Cohn fifth-order function, used to build correlation matrices
  from spatial or temporal distances.

@author: acollet
"""

import numbers
import warnings
from abc import ABC, abstractmethod
from typing import Callable, Optional, Sequence, Union

import numpy as np
from scipy.sparse import csr_matrix, spmatrix

from pyesmda._utils import NDArrayFloat, empirical_cross_covariance, get_anomaly_matrix


class LocalizationStrategy(ABC):
    """Abstract base class defining the interface for localization strategies.

    A localization strategy is responsible for computing a (possibly
    regularized/tapered) empirical cross-covariance matrix between two
    ensembles ``X`` and ``Y``, in order to mitigate spurious correlations
    caused by a finite ensemble size.
    """

    @abstractmethod
    def localize(
        self, X: NDArrayFloat, Y: NDArrayFloat, batch_slice: slice = slice(None)
    ) -> NDArrayFloat:
        """
        Compute the localized empirical cross-covariance matrix of ``X`` and ``Y``.

        Parameters
        ----------
        X : NDArrayFloat
            First ensemble, with members stored along the columns
            (shape ``(n_x, n_ensemble)``).
        Y : NDArrayFloat
            Second ensemble, with members stored along the columns
            (shape ``(n_y, n_ensemble)``).
        batch_slice : slice, optional
            Slice selecting the rows (batch) of the localization matrix to
            use, when only a subset of ``X`` is processed at a time.
            The default is ``slice(None)``, i.e., all rows.

        Returns
        -------
        NDArrayFloat
            The localized cross-covariance matrix, of shape ``(n_x, n_y)``.
        """
        ...

    @abstractmethod
    def localize_multi_dot(
        self,
        X: NDArrayFloat,
        Y: NDArrayFloat,
        *args: NDArrayFloat,
        batch_slice: slice = slice(None),
    ) -> NDArrayFloat:
        """
        Compute the localized product of matrices ``X`` and ``Y`` and optionally
        multiply it with additional matrices.

        This method first computes the localized matrix using :meth:`localize`.
        If additional matrices are provided through ``*args``, they are multiplied
        with the localized matrix using :func:`numpy.linalg.multi_dot` in the order
        ``localized @ args[0] @ args[1] @ ...``.

        Parameters
        ----------
        X : NDArrayFloat
            First input array passed to :meth:`localize`.
        Y : NDArrayFloat
            Second input array passed to :meth:`localize`.
        *args : NDArrayFloat
            Additional matrices to multiply with the localized matrix. Each array
            must have compatible dimensions for matrix multiplication.
        batch_slice : slice, optional
            Slice selecting the rows (batch) of the localization matrix to
            use, when only a subset of ``X`` is processed at a time.
            The default is ``slice(None)``, i.e., all rows.


        Returns
        -------
        NDArrayFloat
            The localized matrix if no additional matrices are provided.
            Otherwise, the result of the chained matrix multiplication
            ``localized @ args[0] @ args[1] @ ...``.
        """
        ...

    def check_localization_shape(
        self, expected_shape: Sequence[int], param_name: str
    ) -> None:
        """
        Check that the localization matrix has the expected shape.

        The base implementation is a no-op; subclasses that hold an
        explicit localization matrix should override it to validate its
        shape and raise a :class:`ValueError` if it does not match.

        Parameters
        ----------
        expected_shape : Sequence[int]
            Expected ``(n_rows, n_cols)`` shape of the localization matrix.
        param_name : str
            Name of the parameter being checked, used in the error message
            if validation fails.

        Returns
        -------
        None
        """
        del expected_shape, param_name  # unused
        pass


class FixedLocalization(LocalizationStrategy):
    r"""
    Localization strategy based on a fixed, user-provided correlation matrix.

    Attributes
    ----------
    correlation_matrix : Optional[Union[sp.sparse.sparray, NDArrayFloat]]
        Correlation matrix based on spatial and temporal distances between
        observations and :math:`\rho_{DD}`. It is used to localize the
        autocovariance matrix of predicted data by applying an elementwise
        multiplication by this matrix.
        Expected dimensions are (:math:`N_{\mathrm{obs}}`, :math:`N_{\mathrm{obs}}`).

    """

    def __init__(
        self,
        correlation_matrix: Optional[Union[NDArrayFloat, spmatrix]] = None,
    ) -> None:
        r"""
        Initialize the instance.

        Parameters
        ----------
        correlation_matrix : Optional[Union[NDArrayFloat, spmatrix]]
            Correlation matrix based on spatial/temporal distances between
            observations/parameters :math:`\rho_{DD}` or :math:`\rho_{MD}`.
            It is used to localize the empirical cross-covariance matrices
            by applying an elementwise multiplication by this matrix.
            Expected dimensions are (:math:`N_{\mathrm{obs}}`, :math:`N_{\mathrm{obs}}`)
            for :math:`\rho_{DD}` and
            (:math:`N_{m}`, :math:`N_{\mathrm{obs}}`) for :math:`\rho_{MD}`. If None, no
            localization is performed. The default is None.
        """

        self.correlation_matrix = (
            csr_matrix(correlation_matrix) if correlation_matrix is not None else None
        )

    def check_localization_shape(
        self, expected_shape: Sequence[int], param_name: str
    ) -> None:
        """
        Check that the stored correlation matrix has the expected shape.

        Parameters
        ----------
        expected_shape : Sequence[int]
            Expected ``(n_rows, n_cols)`` shape of :attr:`correlation_matrix`.
        param_name : str
            Name of the parameter being checked, used in the error message
            if validation fails.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If :attr:`correlation_matrix` is not None and its shape does not
            match ``expected_shape``.
        """
        if self.correlation_matrix is None:
            return
        if self.correlation_matrix.shape != tuple(expected_shape):
            raise ValueError(
                f"{param_name} must be a 2D matrix with "
                f"dimensions {tuple(expected_shape)}."
            )

    def localize(
        self, X: NDArrayFloat, Y: NDArrayFloat, batch_slice: slice = slice(None)
    ) -> NDArrayFloat:
        """
        Compute the empirical cross-covariance of ``X`` and ``Y`` and apply
        localization.

        If :attr:`correlation_matrix` is set, the covariance matrix is
        multiplied elementwise (Schur/Hadamard product) by the rows of
        :attr:`correlation_matrix` selected by ``batch_slice``. Otherwise,
        the raw empirical cross-covariance matrix is returned unchanged.

        Parameters
        ----------
        X : NDArrayFloat
            First ensemble, with members stored along the columns.
        Y : NDArrayFloat
            Second ensemble, with members stored along the columns.
        batch_slice : slice, optional
            Slice selecting the rows of :attr:`correlation_matrix` to use.
            The default is ``slice(None)``, i.e., all rows.

        Returns
        -------
        NDArrayFloat
            The (optionally localized) cross-covariance matrix.
        """
        cov_mat = empirical_cross_covariance(X, Y)
        if self.correlation_matrix is not None:
            return self.correlation_matrix[batch_slice, :].multiply(cov_mat).toarray()
        return cov_mat

    def localize_multi_dot(
        self,
        X: NDArrayFloat,
        Y: NDArrayFloat,
        *args: NDArrayFloat,
        batch_slice: slice = slice(None),
    ) -> NDArrayFloat:
        """
        Compute the localized matrix and optionally multiply it with additional
        matrices.

        This is functionally equivalent to calling :meth:`localize` and then
        chaining the result with ``*args`` via :func:`numpy.linalg.multi_dot`,
        but it avoids recomputing the ensemble anomalies when ``Y is X`` and
        fuses the anomaly-product and localization steps for efficiency.

        Parameters
        ----------
        X : NDArrayFloat
            First ensemble, with members stored along the columns.
        Y : NDArrayFloat
            Second ensemble, with members stored along the columns.
        *args : NDArrayFloat
            Additional matrices to multiply with the localized matrix. Each array
            must have compatible dimensions for matrix multiplication.
        batch_slice : slice, optional
            Slice selecting the rows of :attr:`correlation_matrix` to use.
            The default is ``slice(None)``, i.e., all rows.

        Returns
        -------
        NDArrayFloat
            The localized matrix if no additional matrices are provided.
            Otherwise, the result of the chained matrix multiplication
            ``localized @ args[0] @ args[1] @ ...``.
        """
        X_shift = get_anomaly_matrix(X)
        Y_shift = X_shift if Y is X else get_anomaly_matrix(Y)
        if self.correlation_matrix is not None:
            localized = (
                self.correlation_matrix[batch_slice, :]
                .multiply(X_shift.dot(Y_shift.T))
                .toarray()
            )
            return np.linalg.multi_dot([localized, *args]) if args else localized
        return (
            np.linalg.multi_dot([X_shift, Y_shift.T, *args])
            if args
            else X_shift @ Y_shift.T
        )


class NoLocalization(FixedLocalization):
    """Localization strategy that applies no localization at all.

    Equivalent to :class:`FixedLocalization` with ``correlation_matrix=None``;
    :meth:`~FixedLocalization.localize` simply returns the raw empirical
    cross-covariance matrix. Useful as an explicit "no-op" strategy, e.g. as
    a default value.
    """

    def __init__(self) -> None:
        """Initialize the instance with no correlation matrix."""
        super().__init__()


def default_correlation_threshold(ensemble_size: int) -> float:
    """
    Return the default significance threshold for an adaptive correlation matrix.

    The threshold is computed as ``min(1, max(0, 3 / sqrt(ensemble_size)))``,
    following :cite:t:`luoContinuousHyperparameterOPtimization2022`,
    "Continuous Hyper-parameter OPtimization" (CHOP) in an ensemble Kalman
    filter, Section 2.3, "Localization in the CHOP problem".

    Parameters
    ----------
    ensemble_size : int
        Number of members in the ensemble. Must be strictly positive.

    Returns
    -------
    float
        The correlation threshold, clipped to the range ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``ensemble_size`` is zero.

    Note
    ----
    Original implementation from
    https://github.com/equinor/iterative_ensemble_smoother.

    Examples
    --------
    >>> default_correlation_threshold(1)
    1.0
    >>> default_correlation_threshold(9)
    1.0
    >>> default_correlation_threshold(16)
    0.75
    >>> default_correlation_threshold(36)
    0.5
    >>> default_correlation_threshold(100)
    0.3
    """
    if ensemble_size == 0:
        raise ValueError("The ensemble size cannot be zero!")
    return float(min(1, max(0, 3 / np.sqrt(ensemble_size))))


def cov_to_corr(
    cov_XY: NDArrayFloat,
    stds_X: NDArrayFloat,
    stds_Y: NDArrayFloat,
    inplace: bool = True,
) -> NDArrayFloat:
    """
    Convert a cross-covariance matrix into a cross-correlation matrix.

    Each entry ``cov_XY[i, j]`` is divided by ``stds_X[i] * stds_Y[j]``.
    Divisions by zero (e.g., constant ensemble members with zero standard
    deviation) are replaced with 0, and the resulting values are clipped to
    ``[-1, 1]`` to correct for floating-point round-off; a warning is issued
    if values fall notably outside that range before clipping.

    Parameters
    ----------
    cov_XY : NDArrayFloat
        Cross-covariance matrix of shape ``(n_x, n_y)``.
    stds_X : NDArrayFloat
        Standard deviations associated with the rows of ``cov_XY``, of
        shape ``(n_x,)``.
    stds_Y : NDArrayFloat
        Standard deviations associated with the columns of ``cov_XY``, of
        shape ``(n_y,)``.
    inplace : bool, optional
        If True (default), ``cov_XY`` is modified and returned in place.
        If False, a copy is modified and returned instead, leaving the
        input array untouched.

    Returns
    -------
    NDArrayFloat
        The correlation matrix, with entries in ``[-1, 1]``.

    Note
    ----
    Original implementation from
    https://github.com/equinor/iterative_ensemble_smoother.
    """
    if not inplace:
        cov_XY = cov_XY.copy()
    # Divide each element of cov_XY by the corresponding standard deviations

    cov_XY /= stds_X[:, np.newaxis]
    cov_XY /= stds_Y[np.newaxis, :]

    # divide by zeros,
    # TODO: should not append normally -> stds is not supposed to be null
    cov_XY = np.nan_to_num(cov_XY, nan=0.0)

    # Perform checks and clip values to [-1, 1]
    eps = 1e-8
    if not ((cov_XY.max() <= 1 + eps) and (cov_XY.min() >= -1 - eps)):
        warnings.warn(
            "Cross-correlation matrix has entries not in [-1, 1]."
            f"The min and max values are: {cov_XY.min()} and {cov_XY.max()}"
        )

    return np.clip(cov_XY, a_min=-1, a_max=1, out=cov_XY)


class CorrelationTransform(ABC):
    """Abstract base class for transforms applied to an adaptive correlation matrix.

    Implementations post-process a correlation matrix computed from the
    ensemble (e.g., by thresholding or tempering it) before it is used to
    localize a covariance matrix in :class:`CorrelationBasedLocalization`.
    """

    @abstractmethod
    def __call__(self, correlation_matrix: NDArrayFloat, ne: int) -> NDArrayFloat:
        """
        Transform the given correlation matrix.

        Parameters
        ----------
        correlation_matrix : NDArrayFloat
            Matrix to transform, with entries expected in ``[-1, 1]``.
        ne : int
            Number of members in the ensemble used to compute
            ``correlation_matrix``.

        Returns
        -------
        NDArrayFloat
            The transformed correlation/weight matrix, of the same shape
            as the input.
        """
        ...


def make_correlation_threshold_callable(
    correlation_threshold: Union[Callable[[int], float], float, None],
) -> Callable:
    """
    Normalize a correlation threshold specification into a callable.

    Parameters
    ----------
    correlation_threshold : Union[Callable[[int], float], float, None]
        Either:

        - None, in which case :func:`default_correlation_threshold` is
          returned;
        - a callable with signature ``f(ensemble_size: int) -> float``,
          which is returned unchanged;
        - a float in ``[0, 1]``, which is wrapped into a callable that
          returns that constant value regardless of the ensemble size.

    Returns
    -------
    Callable[[int], float]
        A callable mapping an ensemble size to a correlation threshold.

    Raises
    ------
    TypeError
        If ``correlation_threshold`` is neither None, callable, nor a
        float in ``[0, 1]``.
    """
    # Default value
    if correlation_threshold is None:
        return default_correlation_threshold

    # Create `correlation_threshold` if the argument is a float
    if correlation_threshold is not None:
        if callable(correlation_threshold):
            return correlation_threshold

    # Check the correlation threshold
    if (
        isinstance(correlation_threshold, numbers.Real)
        and correlation_threshold >= 0  # ty:ignore[unsupported-operator]
        and correlation_threshold <= 1
    ):

        def _correlation_threshold(ensemble_size: int) -> float:
            return correlation_threshold  # ty:ignore[invalid-return-type]

        return _correlation_threshold

    raise TypeError("`correlation_threshold` must be a callable or a float in [0, 1]")


class CorrelationThresholding(CorrelationTransform):
    """
    Zero-out correlation entries below a (possibly ensemble-size-dependent) threshold.

    Entries of the correlation matrix whose absolute value does not exceed
    the threshold are set to zero; entries above the threshold are kept
    unchanged. This is a hard-thresholding scheme, as opposed to the
    smooth tapering performed by :class:`CorrelationTempering`.
    """

    __slots__ = ["correlation_threshold"]

    def __init__(
        self, correlation_threshold: Union[Callable[[int], float], float, None] = None
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        correlation_threshold : Union[Callable[[int], float], float, None], optional
            Either a callable with signature f(ensemble_size) -> float, or a
            float in the range [0, 1]. Entries in the covariance matrix that
            are lower than the correlation threshold will be set to zero.
            If None, the default 3/sqrt(ensemble_size) is used. The default is None.
        """
        self.correlation_threshold: Callable[[int], float] = (
            make_correlation_threshold_callable(correlation_threshold)
        )
        assert callable(self.correlation_threshold), (
            "`correlation_threshold` should be callable"
        )

    def __call__(self, correlation_matrix: NDArrayFloat, ne: int) -> NDArrayFloat:
        """
        Zero-out entries of ``correlation_matrix`` below the threshold.

        Parameters
        ----------
        correlation_matrix : NDArrayFloat
            Correlation matrix to threshold.
        ne : int
            Ensemble size, passed to :attr:`correlation_threshold` to
            determine the threshold value.

        Returns
        -------
        NDArrayFloat
            A matrix of the same shape as ``correlation_matrix``, where
            entries with absolute value at or below the threshold have
            been replaced by 0. and all other entries are unchanged.
        """
        return np.where(
            np.abs(correlation_matrix)
            > self.correlation_threshold(correlation_matrix.shape[0]),
            correlation_matrix,
            0.0,
        )


class CorrelationTempering(CorrelationTransform):
    """Apply a user-supplied smooth tempering/tapering function to a correlation matrix.

    Unlike :class:`CorrelationThresholding`, which hard-clips small
    entries to zero, this class delegates entirely to a user-provided
    tempering function, allowing smooth weighting schemes (e.g., the
    Gaspari-Cohn tempering implemented by :func:`gc_correlation_tempering`).
    """

    __slots__ = ["tempering_function"]

    def __init__(
        self, tempering_function: Callable[[NDArrayFloat, int], NDArrayFloat]
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        tempering_function : Callable[[NDArrayFloat, int], NDArrayFloat]
            A callable with signature
            ``f(correlation_matrix: NDArrayFloat, ne: int) -> NDArrayFloat``
            that returns a tempered/tapered version of the input
            correlation matrix, given the ensemble size ``ne``.
        """
        assert callable(tempering_function), "`tempering_function` should be callable"

        # Add as an attribute
        self.tempering_function = tempering_function

    def __call__(self, correlation_matrix: NDArrayFloat, ne: int) -> NDArrayFloat:
        """
        Apply :attr:`tempering_function` to the correlation matrix.

        Parameters
        ----------
        correlation_matrix : NDArrayFloat
            Correlation matrix to temper.
        ne : int
            Ensemble size, forwarded to :attr:`tempering_function`.

        Returns
        -------
        NDArrayFloat
            The tempered correlation/weight matrix returned by
            :attr:`tempering_function`.
        """
        return self.tempering_function(correlation_matrix, ne)


class CorrelationBasedLocalization(LocalizationStrategy):
    """Adaptive localization strategy based on the empirical correlation matrix.

    The empirical cross-covariance of ``X`` and ``Y`` is converted to a
    correlation matrix (via :func:`cov_to_corr`), passed through a
    :class:`CorrelationTransform` (e.g., thresholding or tempering), and
    the resulting weight matrix is used to elementwise-scale the
    covariance matrix.
    """

    __slots__ = ["transform"]

    def __init__(
        self,
        transform: CorrelationTransform,
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        transform : CorrelationTransform
            The transform (thresholding, tempering, ...) applied to the
            empirical correlation matrix before it is used to localize
            the covariance matrix.
        """
        self.transform: CorrelationTransform = transform

    def localize(
        self, X: NDArrayFloat, Y: NDArrayFloat, batch_slice: slice = slice(None)
    ) -> NDArrayFloat:
        """
        Compute the adaptively localized cross-covariance matrix of ``X`` and ``Y``.

        The empirical cross-covariance :math:`C_{SD}` is computed, converted
        to a correlation matrix, passed through :attr:`transform`, and used
        to elementwise-scale :math:`C_{SD}` itself.

        Parameters
        ----------
        X : NDArrayFloat
            First ensemble, with members stored along the columns.
        Y : NDArrayFloat
            Second ensemble, with members stored along the columns.
        batch_slice : slice, optional
            Unused by this strategy; kept for interface compatibility with
            :class:`LocalizationStrategy`.

        Returns
        -------
        NDArrayFloat
            The localized cross-covariance matrix.
        """
        cov_mat = empirical_cross_covariance(X, Y)
        return np.multiply(
            cov_mat,
            self.transform(
                cov_to_corr(
                    cov_mat,
                    np.std(X, axis=1, ddof=1),
                    np.std(Y, axis=1, ddof=1),
                    inplace=False,
                ),
                X.shape[1],
            ),
        )


def localize_multi_dot(
    self,
    X: NDArrayFloat,
    Y: NDArrayFloat,
    *args: NDArrayFloat,
    batch_slice: slice = slice(None),
) -> NDArrayFloat:
    """
    Compute the localized matrix and optionally multiply it with additional matrices.

    Generic implementation of :meth:`LocalizationStrategy.localize_multi_dot`,
    expressed in terms of :meth:`LocalizationStrategy.localize`, suitable
    for mixing into a :class:`LocalizationStrategy` subclass that does not
    provide a more efficient, fused implementation (unlike
    :meth:`FixedLocalization.localize_multi_dot`).

    Parameters
    ----------
    self : LocalizationStrategy
        The localization strategy instance providing a :meth:`localize` method.
    X : NDArrayFloat
        First input array passed to :meth:`~LocalizationStrategy.localize`.
    Y : NDArrayFloat
        Second input array passed to :meth:`~LocalizationStrategy.localize`.
    *args : NDArrayFloat
        Additional matrices to multiply with the localized matrix. Each array
        must have compatible dimensions for matrix multiplication.
    batch_slice : slice, optional
        Slice selecting the batch to process. Currently unused.

    Returns
    -------
    NDArrayFloat
        The localized matrix if no additional matrices are provided.
        Otherwise, the result of the chained matrix multiplication
        ``localized @ args[0] @ args[1] @ ...``.
    """
    localized = self.localize(X, Y)
    return np.linalg.multi_dot([localized, *args]) if args else localized


def _reversed_beta_cumulative(distances: NDArrayFloat, beta: float = 3) -> NDArrayFloat:
    r"""
    Transform distances into weights in ``[0, 1]`` using a beta-like sigmoid function.

    .. math::
        f(d) = 1 - \dfrac{1}{1 + \left(\dfrac{d}{1 - d}\right)^{-\beta}}

    ``f(0) = 0``, ``f(1) = 1``, and the function is monotonically
    increasing over ``d \in [0, 1)``. Values of ``d`` at or above 1 are
    mapped to 1; the value at ``d = 0`` is handled explicitly (would
    otherwise produce ``0 / 0``).

    Parameters
    ----------
    distances : NDArrayFloat
        Input array of distances. Ideally, values should be between 0. and 1.0.
    beta : float, optional
        Shape factor controlling the sharpness of the transition. Must be
        positive or null. The default is 3.0.

    Returns
    -------
    NDArrayFloat
        Array of weights, of the same shape as ``distances``.

    Raises
    ------
    ValueError
        If ``beta`` is negative.
    """
    if beta < 0.0:
        raise ValueError(f"Beta ({beta}) should be positive or null !")

    distances2 = distances.copy()
    distances2[distances == 0] = np.nan
    distances2[distances >= 1] = np.nan
    fact = np.where(
        np.isnan(distances2),
        0.0,
        1.0 / (1.0 + np.power((distances2 / (1 - distances2)), -beta)),
    )
    fact[distances >= 1] = 1.0
    return 1.0 - fact


def gc_correlation_tempering(corr_mat: NDArrayFloat, ne: int) -> NDArrayFloat:
    r"""
    Apply Gaspari-Cohn tempering to a correlation matrix.

    The correlation matrix is first converted into a pseudo-distance,
    ``(1 - |corr_mat|) / (1 - 3 / ne)``, which is then mapped to weights in
    ``[0, 1]`` using the fifth-order Gaspari-Cohn function
    (:func:`distances_to_weights_fifth_order`): entries close to
    perfectly correlated (``|corr| ~ 1``) get a weight close to 1, while
    entries with low correlation are tapered down towards 0.

    See section 2.3, "Localization in the CHOP problem", from
    :cite:t:`luoContinuousHyperparameterOPtimization2022`.

    Parameters
    ----------
    corr_mat : NDArrayFloat
        Correlation matrix, with entries expected in ``[-1, 1]``.
    ne : int
        Number of members in the ensemble used to compute ``corr_mat``.
        Must be strictly greater than 9.

    Returns
    -------
    NDArrayFloat
        Tempering weights, of the same shape as ``corr_mat``, in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``ne <= 9``.
    """
    if ne <= 9:
        raise ValueError("Cannot use the Gaspari-Cohn tempering if Ne <= 9.")
    return distances_to_weights_fifth_order((1 - np.abs(corr_mat)) / (1 - 3 / ne))


def gc_correlation_tempering_positive(corr_mat: NDArrayFloat, ne: int) -> NDArrayFloat:
    r"""
    Apply Gaspari-Cohn tempering to a correlation matrix, zeroing out negative
    correlations.

    Equivalent to :func:`gc_correlation_tempering`, except that weights
    corresponding to negative correlation entries are forced to zero
    instead of being tapered symmetrically with positive correlations.

    See section 2.3, "Localization in the CHOP problem", from
    :cite:t:`luoContinuousHyperparameterOPtimization2022`.

    Parameters
    ----------
    corr_mat : NDArrayFloat
        Correlation matrix, with entries expected in ``[-1, 1]``.
    ne : int
        Number of members in the ensemble used to compute ``corr_mat``.
        Must be strictly greater than 9.

    Returns
    -------
    NDArrayFloat
        Tempering weights, of the same shape as ``corr_mat``, in ``[0, 1]``,
        with entries set to 0 wherever ``corr_mat`` is negative.

    Raises
    ------
    ValueError
        If ``ne <= 9``.
    """
    out = gc_correlation_tempering(corr_mat, ne)
    out[corr_mat < 0] = 0
    return out


def distances_to_weights_beta_cumulative(
    distances: NDArrayFloat, beta: float = 3, scaling_factor: float = 1.0
) -> NDArrayFloat:
    r"""
    Transform distances into weights in ``[0, 1]`` using a scaled beta cumulative
    function.

    This rescales ``distances`` by ``scaling_factor`` and applies
    :func:`_reversed_beta_cumulative`, so that the weight equals 1 at
    distance 0, 0.5 at half the scaling factor, and 0 at (or beyond) the
    scaling factor.

    Parameters
    ----------
    distances : NDArrayFloat
        Input array of distances.
    beta : float, optional
        Shape factor. The smaller ``beta``, the slower the variation; the
        higher ``beta``, the sharper the transition (tends to a Dirac
        function). Must be strictly positive. The default is 3.
    scaling_factor : float, optional
        The scaling factor. At 0, the function equals 1.0, at half the
        scaling factor, it equals 0.5, and at the scaling factor, it
        equals zero. The default is 1.0.

    Returns
    -------
    NDArrayFloat
        Array of weights, of the same shape as ``distances``.

    Raises
    ------
    ValueError
        If ``scaling_factor`` is not strictly positive.
    """
    if scaling_factor <= 0.0:
        raise ValueError(
            f"The scaling factor ({scaling_factor}) should be strictly positive !"
        )
    return _reversed_beta_cumulative(distances / scaling_factor, beta=beta)


def _part1(d: Union[NDArrayFloat, float]) -> Union[NDArrayFloat, float]:
    """
    Evaluate the Gaspari-Cohn fifth-order polynomial for ``0 <= d <= 1``.

    Parameters
    ----------
    d : Union[NDArrayFloat, float]
        Normalized distance(s) (``distance / scaling_factor``), expected in
        ``[0, 1]``.

    Returns
    -------
    Union[NDArrayFloat, float]
        The value(s) of the piecewise polynomial branch used by
        :func:`distances_to_weights_fifth_order` for ``0 <= d <= 1``.
    """
    return -1 / 4 * d**5.0 + 1 / 2 * d**4.0 + 5 / 8 * d**3.0 - 5 / 3 * d**2.0 + 1.0


def _part2(d: Union[NDArrayFloat, float]) -> Union[NDArrayFloat, float]:
    """
    Evaluate the Gaspari-Cohn fifth-order polynomial for ``1 <= d <= 2``.

    Parameters
    ----------
    d : Union[NDArrayFloat, float]
        Normalized distance(s) (``distance / scaling_factor``), expected in
        ``[1, 2]``. Must not contain zeros, since the expression involves
        ``d ** (-1)``.

    Returns
    -------
    Union[NDArrayFloat, float]
        The value(s) of the piecewise polynomial branch used by
        :func:`distances_to_weights_fifth_order` for ``1 <= d <= 2``.
    """
    return (
        1 / 12 * d**5.0
        - 1 / 2 * d**4.0
        + 5 / 8 * d**3
        + 5 / 3 * d**2.0
        - 5.0 * d
        + 4.0
        - 2 / 3 * (d ** (-1.0))
    )


def distances_to_weights_fifth_order(
    distances: NDArrayFloat, scaling_factor: float = 1.0
) -> NDArrayFloat:
    r"""
    Transform distances into weights in ``[0, 1]`` with the Gaspari-Cohn fifth-order
    function.

    .. math::
        f(z) =
            \begin{cases}
            0 & z < 0 \\
            \dfrac{-1}{4} z^{5} + \dfrac{1}{2} z^{4} + \dfrac{5}{8} z^{3} -
            \dfrac{5}{3} z^{2} + 1 & 0 \leq z \leq 1\\
            \dfrac{1}{12} z^{5} - \dfrac{1}{2} z^{4} + \dfrac{5}{8} z^{3} +
            \dfrac{5}{3} z^{2} - 5z + 4 - \dfrac{2}{3} z^{-1} & 1 \leq z \leq 2\\
            0 & z \geq 2
            \end{cases}

    with :math:`z = \dfrac{d}{s}`, :math:`d` the distances, and :math:`s`
    the scaling factor. This is a compactly-supported, smooth taper
    function commonly used for covariance localization.

    See :cite:p:`gaspariConstructionCorrelationFunctions1999`.

    Parameters
    ----------
    distances : NDArrayFloat
        Input distance values.
    scaling_factor : float, optional
        Scaling factor. It is roughly the distance at which weights go
        under 0.25. The default is 1.0.

    Returns
    -------
    NDArrayFloat
        Array of weights, of the same shape as ``distances``, in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``scaling_factor`` is not strictly positive.
    """
    if scaling_factor <= 0:
        raise ValueError(
            f"The scaling_factor ({scaling_factor}) should be strictly positive !"
        )

    distances2 = distances.copy() / scaling_factor

    distances2[distances2 < 0] = np.nan
    distances2[distances2 >= 2.0] = np.nan

    return np.where(
        np.isnan(distances2),
        0.0,
        np.where(
            distances2 >= 1.0,
            _part2(np.where(distances2 <= 0.0, np.nan, distances2)),
            _part1(distances2),
        ),
    )
