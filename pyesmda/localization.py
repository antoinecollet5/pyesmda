"""
Implement some correlation functions.

@author: acollet
"""

import numbers
import warnings
from abc import ABC, abstractmethod
from typing import Callable, Optional, Sequence, Union

import numpy as np
from scipy.sparse import csr_matrix, spmatrix  # type: ignore

from pyesmda.utils import NDArrayFloat, empirical_cross_covariance, get_anomaly_matrix


class LocalizationStrategy(ABC):
    """Abstract class for localization strategy."""

    @abstractmethod
    def localize(
        self, X: NDArrayFloat, Y: NDArrayFloat, batch_slice: slice = slice(None)
    ) -> NDArrayFloat:
        """Apply the localization to the given covariance."""
        ...

    @abstractmethod
    def localize_multi_dot(
        self,
        X: NDArrayFloat,
        Y: NDArrayFloat,
        *args: NDArrayFloat,
        batch_slice: slice = slice(None),
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
        ...

    def check_localization_shape(
        self, expected_shape: Sequence[int], param_name: str
    ) -> None:
        """Check if the localization has the correct shape."""
        del expected_shape, param_name  # unused
        pass


class FixedLocalization(LocalizationStrategy):
    """
    Fixed localization strategy.

    Attributes
    ----------
    correlation_matrix : Optional[csr_matrix]
        Correlation matrix based on spatial and temporal distances between
        observations and :math:`\rho_{DD}`. It is used to localize the
        autocovariance matrix of predicted data by applying an elementwise
        multiplication by this matrix.
        Expected dimensions are (:math:`N_{obs}`, :math:`N_{obs}`).

    """

    def __init__(
        self,
        correlation_matrix: Optional[Union[NDArrayFloat, spmatrix]] = None,
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        correlation_matrix : Optional[Union[NDArrayFloat, spmatrix]]
            Correlation matrix based on spatial/temporal distances between
            observations/parameters :math:`\rho_{DD}` or :math:`\rho_{MD}`.
            It is used to localize the empirical cross-covariance matrices
            by applying an elementwise multiplication by this matrix.
            Expected dimensions are (:math:`N_{obs}`, :math:`N_{obs}`) for
            :math:`\rho_{DD}` and
            (:math:`N_{m}`, :math:`N_{obs}`) for :math:`\rho_{DD}`. It None, no
            localization is performed. The default is None.
        """

        self.correlation_matrix = (
            csr_matrix(correlation_matrix) if correlation_matrix is not None else None
        )

    def check_localization_shape(
        self, expected_shape: Sequence[int], param_name: str
    ) -> None:
        """Check if"""
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
        """Apply the localization to the covariance matrix."""
        cov_mat = empirical_cross_covariance(X, Y)
        if self.correlation_matrix is not None:
            return self.correlation_matrix[batch_slice, :].multiply(cov_mat)  # type: ignore
        return cov_mat

    def localize_multi_dot(
        self,
        X: NDArrayFloat,
        Y: NDArrayFloat,
        *args: NDArrayFloat,
        batch_slice: slice = slice(None),
    ) -> NDArrayFloat:
        """_summary_

        Parameters
        ----------
        X : NDArrayFloat
            _description_
        Y : NDArrayFloat
            _description_

        Returns
        -------
        NDArrayFloat
            _description_
        """
        X_shift = get_anomaly_matrix(X)
        Y_shift = get_anomaly_matrix(Y)
        if self.correlation_matrix is not None:
            return np.linalg.multi_dot(  # type: ignore
                [
                    self.correlation_matrix[batch_slice, :]
                    .multiply(  # type: ignore
                        X_shift.dot(Y_shift.T)  # type: ignore
                    )
                    .toarray(),
                    *args,
                ]
            )
        return np.linalg.multi_dot([X_shift, Y_shift.T, *args])  # type: ignore


class NoLocalization(FixedLocalization):
    """Instance to use when no localization is to be applied."""

    def __init__(self) -> None:
        """Initialize the instance."""
        super().__init__()


def default_correlation_threshold(ensemble_size: int) -> float:
    """
    Return a number that determines whether a correlation is significant.

    Default threshold taken from :cite:t:`luoContinuousHyperparameterOPtimization2022`_,
    Continuous Hyper-parameter OPtimization (CHOP) in an ensemble Kalman filter
    Section 2.3 - Localization in the CHOP problem

    Note
    ----
    Original implementation from https://github.com/equinor/iterative_ensemble_smoother.

    Examples
    --------
    >>> AdaptiveESMDA.correlation_threshold(0)
    1.0
    >>> AdaptiveESMDA.correlation_threshold(9)
    1.0
    >>> AdaptiveESMDA.correlation_threshold(16)
    0.75
    >>> AdaptiveESMDA.correlation_threshold(36)
    0.5
    >>> AdaptiveESMDA.correlation_threshold(100)
    0.3
    """
    return float(min(1, max(0, 3 / np.sqrt(ensemble_size))))


def cov_to_corr(
    cov_XY: NDArrayFloat,
    stds_X: NDArrayFloat,
    stds_Y: NDArrayFloat,
    inplace: bool = True,
) -> NDArrayFloat:
    """
    Convert a covariance matrix to a correlation matrix in-place.

    Note
    ----
    Original implementation from https://github.com/equinor/iterative_ensemble_smoother.
    """
    if not inplace:
        cov_XY = cov_XY.copy()
    # Divide each element of cov_XY by the corresponding standard deviations

    cov_XY /= stds_X[:, np.newaxis]
    cov_XY /= stds_Y[np.newaxis, :]

    # divide by zeros, TODO: should not append normally -> stds is not supposed to be null
    cov_XY = np.nan_to_num(cov_XY, nan=0.0)

    # Perform checks and clip values to [-1, 1]
    eps = 1e-8
    if not ((cov_XY.max() <= 1 + eps) and (cov_XY.min() >= -1 - eps)):
        warnings.warn(
            "Cross-correlation matrix has entries not in [-1, 1]."
            f"The min and max values are: {cov_XY.min()} and {cov_XY.max()}"
        )

    return np.clip(cov_XY, a_min=-1, a_max=1, out=cov_XY)  # type: ignore


class CorrelationTransform(ABC):
    @abstractmethod
    def __call__(self, correlation_matrix: NDArrayFloat, ne: int) -> NDArrayFloat:
        """
        Transform the given correlation matrix.

        Parameters
        ----------
        correlation_matrix : NDArrayFloat
            Matrix to transform.
        ne : int
            Number of members in the ensemble.

        Returns
        -------
        NDArrayFloat
            _description_
        """
        ...


class CorrelationThresholding(CorrelationTransform):
    """Apply a thresholding to the adaptive correlation matrix."""

    __slots__ = ["correlation_threshold"]

    def __init__(
        self, correlation_threshold: Union[Callable[[int], float], float, None] = None
    ) -> None:
        """
        Initialize the instance.

        TODO: add ref.

        Parameters
        ----------
        correlation_threshold : Union[Callable[[int], float], float, None], optional
            Either a callable with signature f(ensemble_size) -> float, or a
            float in the range [0, 1]. Entries in the covariance matrix that
            are lower than the correlation threshold will be set to zero.
            If None, the default 3/sqrt(ensemble_size) is used. The default is None.
        """
        # Check the correlation threshold
        is_float = (
            isinstance(correlation_threshold, numbers.Real)
            and correlation_threshold >= 0
            and correlation_threshold <= 1
        )

        # Create `correlation_threshold` if the argument is a float
        if callable(correlation_threshold):
            _correlation_threshold: Callable[[int], float] = correlation_threshold
        elif is_float:
            corr_threshold: float = correlation_threshold  # type: ignore

            def _correlation_threshold(ensemble_size: int) -> float:
                return corr_threshold

        elif correlation_threshold is not None:
            raise TypeError(
                "`correlation_threshold` must be a callable or a float in [0, 1]"
            )
        else:
            _correlation_threshold = default_correlation_threshold

        assert callable(
            _correlation_threshold
        ), "`correlation_threshold` should be callable"

        # Add as an attribute
        self.correlation_threshold: Callable[[int], float] = _correlation_threshold

    def __call__(self, correlation_matrix: NDArrayFloat, ne: int) -> NDArrayFloat:
        """Transform the correlation matrix."""
        return np.where(
            np.abs(correlation_matrix)
            > self.correlation_threshold(correlation_matrix.shape[0]),
            correlation_matrix,
            0.0,
        )


class CorrelationTempering(CorrelationTransform):
    """Apply a tempring to the adaptive correlation matrix."""

    __slots__ = ["correlation_threshold"]

    def __init__(
        self, tempering_function: Callable[[NDArrayFloat, int], NDArrayFloat]
    ) -> None:
        """
        Initialize the instance.

        TODO: add ref.

        Parameters
        ----------
        correlation_threshold : Union[Callable[[int], float], float, None], optional
            Either a callable with signature f(ensemble_size) -> float, or a
            float in the range [0, 1]. Entries in the covariance matrix that
            are lower than the correlation threshold will be set to zero.
            If None, the default 3/sqrt(ensemble_size) is used. The default is None.
        """
        assert callable(
            tempering_function
        ), "`correlation_threshold` should be callable"

        # Add as an attribute
        self.tempering_function = tempering_function

    def __call__(self, correlation_matrix: NDArrayFloat, ne: int) -> NDArrayFloat:
        """Transform the correlation matrix."""
        return self.tempering_function(correlation_matrix, ne)


class CorrelationBasedLocalization(LocalizationStrategy):
    """Implement an adaptative correlation based localization strategy."""

    __slots__ = ["correlation_threshold"]

    def __init__(
        self,
        transform: CorrelationTransform,
    ) -> None:
        """
        Initialize the instance.

        TODO: add ref.

        """
        self.transform: CorrelationTransform = transform

    def localize(
        self, X: NDArrayFloat, Y: NDArrayFloat, batch_slice: slice = slice(None)
    ) -> NDArrayFloat:
        """Apply the localization to C_SD."""
        cov_mat = empirical_cross_covariance(X, Y)
        return np.multiply(
            cov_mat,
            self.transform(
                np.abs(
                    cov_to_corr(
                        cov_mat,
                        np.std(X, axis=1, ddof=1),
                        np.std(Y, axis=1, ddof=1),
                        inplace=False,
                    )
                ),
                X.shape[0],
            ),
        )

    def localize_multi_dot(
        self,
        X: NDArrayFloat,
        Y: NDArrayFloat,
        *args: NDArrayFloat,
        batch_slice: slice = slice(None),
    ) -> NDArrayFloat:
        """_summary_

        Parameters
        ----------
        X : NDArrayFloat
            _description_
        Y : NDArrayFloat
            _description_
        *args : NDArrayFloat
            _description_

        Returns
        -------
        NDArrayFloat
            _description_
        """
        return np.linalg.multi_dot(  # type: ignore
            [
                self.localize(X, Y),
                *args,
            ]
        )


def _reversed_beta_cumulative(distances: NDArrayFloat, beta: float = 3) -> NDArrayFloat:
    r"""
    Transform the distances into weights between 0 and 1 with a beta function.

    .. math::
        1 - \dfrac{1}{1 + \left(\dfrac{d}{1 - d}\right)^{-\beta}}

    Parameters
    ----------
    x : NDArrayFloat
        Input array. Idealy, values should be between 0. and 1.0.
    beta: float, optional
        Shape factor. Must be positive or null. The default is 3.0.

    Returns
    -------
    NDArrayFloat.
        Array of same dimension as input array.

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
    """
    Apply the Gaspari-Cohn tempering to the correlation matrix.

    See section 2.3, Localization in the CHOP problem from
    :cite:t:`ContinuousHyperparameterOPtimization2022`.

    .. math::
        (TODO) 1 - \dfrac{1}{1 + \left(\dfrac{d}{s - d}\right)^{-\beta}}
    """
    if ne <= 9:
        raise ValueError("Cannot use the Gaspari-Cohn tempering if Ne <= 9.")
    return distances_to_weights_fifth_order((1 - np.abs(corr_mat)) / (1 - 3 / ne))


def distances_to_weights_beta_cumulative(
    distances: NDArrayFloat, beta: float = 3, scaling_factor: float = 1.0
) -> NDArrayFloat:
    r"""
    Transform the distances into weights between 0 and 1 with a beta function.

    .. math::
        1 - \dfrac{1}{1 + \left(\dfrac{d}{s - d}\right)^{-\beta}}

    Parameters
    ----------
    distances : NDArrayFloat
        Input array of distances.
    beta: float, optional
        Shape factor. The smalest beta, the slower the variation, the higher beta
        the sharpest the transition (tends to a dirac function). Must be strictly
        positive. The default is 3.
    scaling_factor: float, optional
        The scaling factor. At 0, the function equals 1.0, at half the scaling factor,
        it equals 0.5, and at the scaling factor, is equals zero.
        The default is 1.0.

    Returns
    -------
    NDArrayFloat.
        Array of same dimension as input array.

    """
    if scaling_factor <= 0.0:
        raise ValueError(
            f"The scaling factor ({scaling_factor}) should be strictly positive !"
        )
    return _reversed_beta_cumulative(distances / scaling_factor, beta=beta)


def _part1(d: Union[NDArrayFloat, float]) -> Union[NDArrayFloat, float]:
    return (
        -1 / 4 * d**5.0 + 1 / 2 * d**4.0 + 5 / 8 * d**3.0 - 5 / 3 * d**2.0 + 1.0
    )


def _part2(d: Union[NDArrayFloat, float]) -> Union[NDArrayFloat, float]:
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
    Transform the distances into weights between 0 and 1 with a fifth order function.

    .. math::
        f(z) =
            \begin{cases}
            0 & z < 0 \\
            \dfrac{-1}{4} z^{5} + \dfrac{1}{2} z^{4} + \dfrac{5}{8} z^{3} -
            \dfrac{5}{3} z^{2} + 1 & 0 \leq z \leq 1\\
            \dfrac{1}{12} z^{5} - \dfrac{1}{2} z^{4} + \dfrac{5}{8} z^{3} +
            \dfrac{5}{3} z^{2} - 5z + 4 - \dfrac{2}{3} z^{-1} & 1 \leq z \leq 2\\
            \end{cases}

    with :math:`z = \dfrac{d}{s}`,  :math:`d` the distances, and :math:`s`
    the scaling factor.

    See :cite:p:`gaspariConstructionCorrelationFunctions1999`.

    Parameters
    ----------
    distances : NDArrayFloat
        Input distances values.
    scaling_factor: float, optional
        Scaling factor. It is roughly the distance at which weights go under 0.25.
        The default is 1.0.

    Returns
    -------
    NDArrayFloat.
        Array of same dimension as input array.

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
            _part2(np.where(distances2 <= 0.0, np.nan, distances2)),  # type: ignore
            _part1(distances2),
        ),
    )
