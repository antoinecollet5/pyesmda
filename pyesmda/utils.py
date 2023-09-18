"""
Implement the ES-MDA algorithms.

@author: acollet
"""
from functools import lru_cache, wraps
from typing import List, Union

import numpy as np
import numpy.typing as npt
import scipy as sp  # type: ignore
from scipy.sparse import csr_matrix  # type: ignore
from scipy.sparse.linalg import gmres  # type: ignore

NDArrayFloat = npt.NDArray[np.float64]


def np_cache(*args, **kwargs):
    """
    LRU cache implementation for functions whose FIRST parameter is a numpy array.

    Examples
    --------
    >>> array = np.array([[1, 2, 3], [4, 5, 6]])
    >>> @np_cache(maxsize=256)
    ... def multiply(array, factor):
    ...     print("Calculating...")
    ...     return factor*array
    >>> multiply(array, 2)
    Calculating...
    array([[ 2,  4,  6],
           [ 8, 10, 12]])
    >>> multiply(array, 2)
    array([[ 2,  4,  6],
           [ 8, 10, 12]])
    >>> multiply.cache_info()
    CacheInfo(hits=1, misses=1, maxsize=256, currsize=1)

    """

    def decorator(function):
        @wraps(function)
        def wrapper(np_array, *args, **kwargs):
            hashable_array = array_to_tuple(np_array)
            return cached_wrapper(hashable_array, *args, **kwargs)

        @lru_cache(*args, **kwargs)
        def cached_wrapper(hashable_array, *args, **kwargs):
            array = np.array(hashable_array)
            return function(array, *args, **kwargs)

        def array_to_tuple(np_array):
            """Iterates recursively."""
            try:
                return tuple(array_to_tuple(_) for _ in np_array)
            except TypeError:
                return np_array

        # copy lru_cache attributes over too
        wrapper.cache_info = cached_wrapper.cache_info
        wrapper.cache_clear = cached_wrapper.cache_clear

        return wrapper

    return decorator


def get_ensemble_variance(
    ensemble: NDArrayFloat,
) -> NDArrayFloat:
    """
    Get the given ensemble variance (diagonal terms of the covariance matrix).

    Parameters
    ----------
    ensemble : NDArrayFloat
        Ensemble of realization with diemnsions (:math:`N_{e}, N_{m}`),
        $N_{m}$), $N_{e}$ and $N_{m}$
        being the ensemble size and one member size respectively.

    Returns
    -------
    NDArrayFloat
        The variance as a 1d array.

    Raises
    ------
    ValueError
        If the ensemble is not a 2D matrix.
    """
    # test ensemble size
    if len(ensemble.shape) != 2:
        raise ValueError("The ensemble must be a 2D matrix!")
    return np.sum(((ensemble - np.mean(ensemble, axis=0)) ** 2), axis=0) / (
        ensemble.shape[0] - 1.0
    )  # type: ignore


@np_cache()
def get_anomaly_matrix(
    ensemble: NDArrayFloat,
) -> NDArrayFloat:
    """
    Return the zero-mean (i.e., centered) anomaly matrix of the ensemble.

    Parameters
    ----------
    ensemble: NDArrayFloat
        Ensemble of realization with shape ($N_{e}$, $N_{m}$), $N_{e}$ and $N_{m}$
        being the ensemble size and one member size respectively.

    Examples
    --------
    >>> X = np.array([[-2.4, -0.3,  0.7,  0.2,  1.1],
    ...               [-1.5,  0.4, -0.4, -0.9,  1. ],
    ...               [-0.1, -0.4, -0. , -0.5,  1.1]])
    >>> get_anomaly_matrix(X)
    array([[-0.75424723, -0.11785113,  0.87209836],
        [-0.14142136,  0.35355339, -0.21213203],
        [ 0.42426407, -0.35355339, -0.07071068],
        [ 0.42426407, -0.35355339, -0.07071068],
        [ 0.02357023, -0.04714045,  0.02357023]])

    Return
    ------
    The anomaly matrix with with shape ($N_{e}$, $N_{m}$).
    """
    return (ensemble.T - np.mean(ensemble.T, axis=1, keepdims=True)) / np.sqrt(
        ensemble.shape[0] - 1  # type: ignore
    )


def empirical_covariance_upper(ensemble: NDArrayFloat) -> NDArrayFloat:
    """Compute the upper triangular part of the empirical covariance matrix X.

    The output shape (num_variables, num_observations).

    Parameter
    ---------
    ensemble: NDArrayFloat
        Ensemble of values.

    Examples
    --------
    >>> X = np.array([[-2.4, -0.3,  0.7,  0.2,  1.1],
    ...               [-1.5,  0.4, -0.4, -0.9,  1. ],
    ...               [-0.1, -0.4, -0. , -0.5,  1.1]])
    >>> empirical_covariance_upper(X.T)
    array([[1.873, 0.981, 0.371],
           [0.   , 0.997, 0.392],
           [0.   , 0.   , 0.407]])

    Naive computation:

    >>> approximate_covariance_matrix_from_ensembles(X.T, X.T)
    array([[1.873, 0.981, 0.371],
           [0.981, 0.997, 0.392],
           [0.371, 0.392, 0.407]])
    """
    # https://www.math.utah.edu/software/lapack/lapack-blas/dsyrk.html
    XXT: npt.NDArray[np.double] = sp.linalg.blas.dsyrk(
        alpha=1.0, a=get_anomaly_matrix(ensemble)
    )
    return XXT


def approximate_covariance_matrix_from_ensembles(
    ensemble_1: NDArrayFloat, ensemble_2: NDArrayFloat
) -> NDArrayFloat:
    r"""
    Approximate the covariance matrix between two ensembles in the EnKF way.

    The covariance matrice :math:`C_{m1m2}`
    is approximated from the ensemble in the standard way of EnKF
    :cite:p:`evensenDataAssimilationEnsemble2007,aanonsenEnsembleKalmanFilter2009`:

    .. math::
        C_{p1p2} = \frac{1}{N_{e} - 1} \sum_{j=1}^{N_{e}}\left(m1_{j} -
        \overline{m1}\right)\left(m2_{j}
        - \overline{m2} \right)^{T}

    Parameters
    ----------
    ensemble_1 : NDArrayFloat
        First ensemble of realization with diemnsions (:math:`N_{e}, N_{m1}`).
    ensemble_2 : NDArrayFloat
        Second ensemble of realization with diemnsions (:math:`N_{e}, N_{m2}`).

    Returns
    -------
    NDArrayFloat
        The two ensembles approximated covariance matrix.

    Examples
    --------
    >>> X = np.array([[-2.4, -0.3,  0.7],
    ...               [ 0.2,  1.1, -1.5]])
    >>> Y = np.array([[ 0.4, -0.4, -0.9],
    ...               [ 1. , -0.1, -0.4],
    ...               [-0. , -0.5,  1.1],
    ...               [-1.8, -1.1,  0.3]])
    >>> approximate_covariance_matrix_from_ensembles(X.T, Y.T)
    array([[-1.035     , -1.15833333,  0.66      ,  1.56333333],
           [ 0.465     ,  0.36166667, -1.08      , -1.09666667]])

    Verify against numpy.cov

    >>> np.cov(X, rowvar=True, ddof=1)
    array([[ 2.50333333, -0.99666667],
           [-0.99666667,  1.74333333]])
    >>> approximate_covariance_matrix_from_ensembles(X.T, X.T)
    array([[ 2.50333333, -0.99666667],
           [-0.99666667,  1.74333333]])

    Raises
    ------
    ValueError
        _description_
    """
    # test ensemble size
    if (
        ensemble_1.shape[0] != ensemble_2.shape[0]  # type: ignore
        or len(ensemble_1.shape) != 2
        or len(ensemble_2.shape) != 2
    ):
        raise ValueError(
            "The ensemble should be 2D matrices with equal first dimension!"
        )
    return get_anomaly_matrix(ensemble_1) @ get_anomaly_matrix(ensemble_2).T


# define an alias
empirical_cross_covariance = approximate_covariance_matrix_from_ensembles


def approximate_cov_mm(ensemble: NDArrayFloat) -> NDArrayFloat:
    r"""
    Approximate the parameters autocovariance matrix from the ensemble.

    The covariance matrice :math:`C^{l}_{MM}`
    is approximated from the ensemble in the standard way of EnKF
    :cite:p:`evensenDataAssimilationEnsemble2007,aanonsenEnsembleKalmanFilter2009`:

    .. math::
        C^{l}_{MM} = \frac{1}{N_{e} - 1} \sum_{j=1}^{N_{e}}\left(m^{l}_{j} -
        \overline{m^{l}}\right)\left(m^{l}_{j}
        - \overline{m^{l}} \right)^{T}

    with :math:`\overline{m^{l}}`, the parameters
    ensemble means, at iteration :math:`l`.

    Parameters
    ----------
    ensemble: NDArrayFloat
        Ensemble of parameters realization with diemnsions (:math:`N_{e}, N_{m}`).
    """
    return approximate_covariance_matrix_from_ensembles(ensemble, ensemble)


def compute_ensemble_average_normalized_objective_function(
    pred_ensemble: NDArrayFloat,
    obs: NDArrayFloat,
    cov_obs: Union[NDArrayFloat, csr_matrix],
) -> float:
    r"""
    Compute the ensemble average normalized objective function.

    .. math::

        \overline{O}_{N_{d}} = \frac{1}{N_{e}} \sum_{j=1}^{N_{e}} O_{N_{d}, j}

    .. math::

        \textrm{with  } O_{N_{d}, j} = \frac{1}{2N_{d}}
        \sum_{j=1}^{N_{e}}\left(d^{l}_{j}
        - {d_{obs}} \right)^{T}C_{D}^{-1}\left(d^{l}_{j}
        - {d_{obs}} \right)\\

    Parameters
    ----------
    pred_ensemble : NDArrayFloat
        Vector of predicted values.
    obs : NDArrayFloat
        Vector of observed values.
    cov_obs : Union[NDArrayFloat, csr_matrix]
        Covariance matrix of observed data measurement errors with dimensions
        (:math:`N_{obs}`, :math:`N_{obs}`). Also denoted :math:`R`. This can be a
        sparse matrix.

    Returns
    -------
    float
        The objective function.

    """

    def member_obj_fun(pred: NDArrayFloat) -> float:
        return compute_normalized_objective_function(pred, obs, cov_obs)

    return np.mean(
        list(
            map(
                member_obj_fun,
                pred_ensemble,
            )
        )  # type: ignore
    )


def compute_normalized_objective_function(
    pred: NDArrayFloat,
    obs: NDArrayFloat,
    cov_obs: Union[NDArrayFloat, csr_matrix],
) -> float:
    r"""
    Compute the normalized objective function for a given member :math:`j`.

    .. math::

        O_{N_{d}, j} = \frac{1}{2N_{d}} \sum_{j=1}^{N_{e}}\left(d^{l}_{j}
        - {d_{obs}} \right)^{T}C_{D}^{-1}\left(d^{l}_{j}
        - {d_{obs}} \right)

    Parameters
    ----------
    pred : NDArrayFloat
        Vector of predicted values.
    obs : NDArrayFloat
        Vector of observed values.
    cov_obs : Union[NDArrayFloat, csr_matrix]
        Covariance matrix of observed data measurement errors with dimensions
        (:math:`N_{obs}`, :math:`N_{obs}`). Also denoted :math:`R`. This can be a
        sparse matrix.

    Returns
    -------
    float
        The objective function.

    """
    residuals: NDArrayFloat = obs - pred
    try:
        # case of dense array
        return (
            1
            / (2 * obs.size)
            * np.dot(residuals.T, np.linalg.solve(cov_obs, residuals))
        )
    except Exception:
        # case of sparse matrices
        return (
            1
            / (2 * obs.size)
            * np.dot(residuals.T, gmres(cov_obs, residuals, atol=1e-15)[0])
        )


def inflate_ensemble_around_its_mean(
    ensemble: NDArrayFloat, inflation_factor: float
) -> NDArrayFloat:
    r"""
    Inflate the given parameter ensemble around its mean.

    .. math::
        m^{l+1}_{j} \leftarrow r^{l+1}\left(m^{l+1}_{j} - \frac{1}{N_{e}}
        \sum_{j}^{N_{e}}m^{l+1}_{j}\right)
        + \frac{1}{N_{e}}\sum_{j}^{N_{e}}m^{l+1}_{j}

    Parameters
    ----------
    ensemble: NDArrayFloat
        Ensemble of realization with diemnsions (:math:`N_{e}, N_{m}`).

    Returns
    -------
    NDArrayFloat
        The inflated ensemble.
    """
    if not inflation_factor == 1.0:
        return inflation_factor * (ensemble - np.mean(ensemble, axis=0)) + np.mean(
            ensemble, axis=0
        )
    return ensemble


def check_nans_in_predictions(d_pred: NDArrayFloat, assimilation_step: int) -> None:
    """
    Check and raise an exception if there is any NaNs in the input predictions array.

    Parameters
    ----------
    d_pred : NDArrayFloat
        Input prediction vector(s).
    assimilation_step : int
        Assimilation step index. 0 means before the first assimilation.

    Raises
    ------
    Exception
        Raised if NaNs are found. It indicates which ensemble members have incorrect
        predictions, and at which assimilaiton step.
    """
    if not np.isnan(d_pred).any():
        return

    # indices of members for which nan have been found
    error_indices: List[int] = sorted(set(np.where(np.isnan(d_pred))[0]))
    if assimilation_step == 0:
        msg: str = "with the initial ensemble predictions "
    else:
        msg = f" after assimilation step {assimilation_step}"
    raise Exception(
        "Something went wrong " + msg + " -> NaN values"
        f" are found in predictions for members {error_indices} !"
    )
