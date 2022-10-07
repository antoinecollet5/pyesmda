"""
Implement the ES-MDA algorithms.

@author: acollet
"""
from typing import List

import numpy as np
import numpy.typing as npt


def approximate_covariance_matrix_from_ensembles(
    ensemble_1: npt.NDArray[np.float64], ensemble_2: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
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
    ensemble_1 : npt.NDArray[np.float64]
        First ensemble of realization with diemnsions (:math:`N_{e}, N_{m1}`).
    ensemble_2 : npt.NDArray[np.float64]
        Second ensemble of realization with diemnsions (:math:`N_{e}, N_{m2}`).

    Returns
    -------
    npt.NDArray[np.float64]
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
    delta_m1: npt.NDArray[np.float64] = ensemble_1 - np.mean(ensemble_1, axis=0)
    delta_m2: npt.NDArray[np.float64] = ensemble_2 - np.mean(ensemble_2, axis=0)

    cov: npt.NDArray[np.float64] = np.zeros((ensemble_1.shape[1], ensemble_2.shape[1]))

    for j in range(ensemble_1.shape[0]):
        cov += np.outer(delta_m1[j, :], delta_m2[j, :])

    return cov / (ensemble_1.shape[0] - 1.0)


def approximate_cov_mm(m_ensemble: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
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
    m_ensemble: npt.NDArray[np.float64]
        Ensemble of parameters realization with diemnsions (:math:`N_{e}, N_{m}`).
    """
    return approximate_covariance_matrix_from_ensembles(m_ensemble, m_ensemble)


def compute_ensemble_average_normalized_objective_function(
    pred_ensemble: npt.NDArray[np.float64],
    obs: npt.NDArray[np.float64],
    cov_obs: npt.NDArray[np.float64],
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
    pred_ensemble : npt.NDArray[np.float64]
        Vector of predicted values.
    obs : npt.NDArray[np.float64]
        Vector of observed values.
    cov_obs : npt.NDArray[np.float64]
        Covariance matrix of observed data measurement errors with dimensions
        (:math:`N_{obs}`, :math:`N_{obs}`). Also denoted :math:`R`.

    Returns
    -------
    float
        The objective function.

    """

    def member_obj_fun(pred: npt.NDArray[np.float64]) -> float:
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
    pred: npt.NDArray[np.float64],
    obs: npt.NDArray[np.float64],
    cov_obs: npt.NDArray[np.float64],
) -> float:
    r"""
    Compute the normalized objective function for a given member :math:`j`.

    .. math::

        O_{N_{d}, j} = \frac{1}{2N_{d}} \sum_{j=1}^{N_{e}}\left(d^{l}_{j}
        - {d_{obs}} \right)^{T}C_{D}^{-1}\left(d^{l}_{j}
        - {d_{obs}} \right)

    Parameters
    ----------
    pred : npt.NDArray[np.float64]
        Vector of predicted values.
    obs : npt.NDArray[np.float64]
        Vector of observed values.
    cov_obs : npt.NDArray[np.float64]
        Covariance matrix of observed data measurement errors with dimensions
        (:math:`N_{obs}`, :math:`N_{obs}`). Also denoted :math:`R`.

    Returns
    -------
    float
        The objective function.

    """
    residuals: npt.NDArray[np.float64] = obs - pred
    return 1 / (2 * obs.size) * np.dot(residuals.T, np.linalg.solve(cov_obs, residuals))


def inflate_ensemble_around_its_mean(
    ensemble: npt.NDArray[np.float64], inflation_factor: float
) -> npt.NDArray[np.float64]:
    r"""
    Inflate the given parameter ensemble around its mean.

    .. math::
        m^{l+1}_{j} \leftarrow r^{l+1}\left(m^{l+1}_{j} - \frac{1}{N_{e}}
        \sum_{j}^{N_{e}}m^{l+1}_{j}\right)
        + \frac{1}{N_{e}}\sum_{j}^{N_{e}}m^{l+1}_{j}

    Parameters
    ----------
    ensemble: npt.NDArray[np.float64]
        Ensemble of realization with diemnsions (:math:`N_{e}, N_{m}`).

    Returns
    -------
    npt.NDArray[np.float64]
        The inflated ensemble.
    """
    if not inflation_factor == 1.0:
        return inflation_factor * (ensemble - np.mean(ensemble, axis=0)) + np.mean(
            ensemble, axis=0
        )
    return ensemble


def check_nans_in_predictions(
    d_pred: npt.NDArray[np.float64], assimilation_step: int
) -> None:
    """
    Check and raise an exception if there is any NaNs in the input predictions array.

    Parameters
    ----------
    d_pred : npt.NDArray[np.float64]
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
