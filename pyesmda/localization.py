"""
Implement some correlation functions.

@author: acollet
"""
import numpy as np

from pyesmda.utils import NDArrayFloat


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
    fact[distances >= 1] = 1
    return 1 - fact


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


def _part1(d: float) -> float:
    return (
        -1 / 4 * d**5.0 + 1 / 2 * d**4.0 + 5 / 8 * d**3.0 - 5 / 3 * d**2.0 + 1.0
    )


def _part2(d: float) -> float:
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
            \dfrac{-1}{4} z^{5} + \dfrac{1}{2} z^{4} + \dfrac{5}{8} z^{3} - \dfrac{5}{3} z^{2} + 1 & 0 \leq z \leq 1\\
            \dfrac{1}{12} z^{5} - \dfrac{1}{2} z^{4} + \dfrac{5}{8} z^{3} + \dfrac{5}{3} z^{2} - 5z + 4 - \dfrac{2}{3} z^{-1} & 1 \leq z \leq 2\\
            \end{cases}

    with :math:`z = \dfrac{d}{s}`,  :math:`d` the distances, and :math:`s` the scaling factor.

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
            _part2(np.where(distances2 <= 0.0, np.nan, distances2)),
            _part1(distances2),
        ),
    )
