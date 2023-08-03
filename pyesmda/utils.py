"""
Implement the ES-MDA algorithms.

@author: acollet
"""
from typing import List, Union

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import gmres

NDArrayFloat = npt.NDArray[np.float64]


def get_ensemble_variance(
    m_ensemble: NDArrayFloat,
) -> NDArrayFloat:
    """
    Get the given ensemble variance (diagonal terms of the covariance matrix).

    Parameters
    ----------
    m_ensemble : NDArrayFloat
        Ensemble of realization with diemnsions (:math:`N_{e}, N_{m1}`).

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
    if len(m_ensemble.shape) != 2:
        raise ValueError("The ensemble must be a 2D matrix!")
    return np.sum(((m_ensemble - np.mean(m_ensemble, axis=0)) ** 2), axis=0) / (
        m_ensemble.shape[0] - 1.0
    )


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

    Raises
    ------
    ValueError
        _description_
    """
    # test ensemble size
    if (
        ensemble_1.shape[0] != ensemble_2.shape[0]
        or len(ensemble_1.shape) != 2
        or len(ensemble_2.shape) != 2
    ):
        raise ValueError(
            "The ensemble should be 2D matrices with equal first dimension!"
        )

    # Delta with average per ensemble member
    delta_m1: NDArrayFloat = ensemble_1 - np.mean(ensemble_1, axis=0)
    delta_m2: NDArrayFloat = ensemble_2 - np.mean(ensemble_2, axis=0)

    cov: NDArrayFloat = np.zeros((ensemble_1.shape[1], ensemble_2.shape[1]))

    for j in range(ensemble_1.shape[0]):
        cov += np.outer(delta_m1[j, :], delta_m2[j, :])

    return cov / (ensemble_1.shape[0] - 1.0)


def approximate_cov_mm(m_ensemble: NDArrayFloat) -> NDArrayFloat:
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
    m_ensemble: NDArrayFloat
        Ensemble of parameters realization with diemnsions (:math:`N_{e}, N_{m}`).
    """
    return approximate_covariance_matrix_from_ensembles(m_ensemble, m_ensemble)


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
        )
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
        msg: str = f" after assimilation step {assimilation_step}"
    raise Exception(
        "Something went wrong " + msg + " -> NaN values"
        f" are found in predictions for members {error_indices} !"
    )
