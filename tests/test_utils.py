"""
Test for the Ensemble-Smoother with Multiple Data Assimilation utils.

@author: acollet
"""

from contextlib import contextmanager

import numpy as np
import pytest
import scipy as sp

from pyesmda.utils import (
    approximate_covariance_matrix_from_ensembles,
    check_nans_in_predictions,
    empirical_covariance_upper,
    empirical_cross_covariance,
    get_anomaly_matrix,
    get_ensemble_variance,
    inflate_ensemble_around_its_mean,
    ls_cost_function,
)


@contextmanager
def does_not_raise():
    yield


def test_get_anomaly_matrix() -> None:
    np.testing.assert_allclose(
        get_anomaly_matrix(
            np.array(
                [
                    [-2.4, -0.3, 0.7, 0.2, 1.1],
                    [-1.5, 0.4, -0.4, -0.9, 1.0],
                    [-0.1, -0.4, -0.0, -0.5, 1.1],
                ],
                dtype=np.float64,
            ).T
        ),
        np.array(
            [
                [-0.75424723, -0.11785113, 0.87209836],
                [-0.14142136, 0.35355339, -0.21213203],
                [0.42426407, -0.35355339, -0.07071068],
                [0.42426407, -0.35355339, -0.07071068],
                [0.02357023, -0.04714045, 0.02357023],
            ]
        ),
        rtol=1e-6,
        atol=1e-6,
    )


def test_empirical_covariance_upper() -> None:
    X = np.array(
        [
            [-2.4, -0.3, 0.7, 0.2, 1.1],
            [-1.5, 0.4, -0.4, -0.9, 1.0],
            [-0.1, -0.4, -0.0, -0.5, 1.1],
        ]
    )
    empirical_covariance_upper(X)

    np.triu(empirical_cross_covariance(X, X))


@pytest.mark.parametrize(
    "pred, obs, cov_obs, expected, exception",
    [
        (
            np.ones((20)),
            np.ones((20)) * 2.0,
            sp.linalg.cholesky(np.diag((np.ones((20)) * 0.5) ** 2), lower=False),
            20.0,
            does_not_raise(),
        ),
        (
            np.ones((20)),
            np.ones((20)) * 2.0,
            sp.linalg.cholesky(np.diag((np.ones((20)) * 2.0) ** 2), lower=False),
            5.0,
            does_not_raise(),
        ),
        (
            np.ones((20, 10)),
            np.ones((20)) * 2.0,
            sp.linalg.cholesky(np.diag((np.ones((20)) * 0.5) ** 2), lower=False),
            np.ones(10) * 20.0,
            does_not_raise(),
        ),
        (
            np.ones((20, 10)),
            np.ones((20)) * 2.0,
            np.sqrt(np.ones((20)) * 2),
            np.ones(10) * 5.0,
            does_not_raise(),
        ),
        (
            np.ones((20, 10)),
            np.ones((20)) * 2.0,
            np.sqrt(np.ones((20, 20, 20)) * 2),
            np.ones(20) * 5.0,
            pytest.raises(
                ValueError, match="cov_obs_cholesky must be a 2D array or a 1D array."
            ),
        ),
    ],
)
def test_normalized_objective_function(pred, obs, cov_obs, expected, exception) -> None:
    with exception:
        np.testing.assert_allclose(ls_cost_function(pred, obs, cov_obs), expected)


@pytest.mark.parametrize(
    "ens1,ens2,expected,expected_exception",
    [
        (  # simple construction
            np.ones((30, 20)),
            np.ones((10, 20)),
            np.zeros((30, 10)),
            does_not_raise(),
        ),
        (  # issue with stdev_d
            np.ones((30, 10)),
            np.ones((10, 21)),
            None,
            pytest.raises(
                ValueError,
                match=(
                    r"The ensemble should be 2D matrices with equal second dimension!"
                ),
            ),
        ),
        (  # issue with stdev_d
            np.ones(10),
            np.ones((10, 10)),
            None,
            pytest.raises(
                ValueError,
                match=(
                    r"The ensemble should be 2D matrices with equal second dimension!"
                ),
            ),
        ),
    ],
)
def test_approximate_covariance_matrix_from_ensembles(
    ens1, ens2, expected, expected_exception
) -> None:
    with expected_exception:
        np.testing.assert_almost_equal(
            approximate_covariance_matrix_from_ensembles(ens1, ens2), expected
        )


def test_approximate_covariance_matrix_from_ensembles_res() -> None:
    X = np.array([[-2.4, -0.3, 0.7], [0.2, 1.1, -1.5]])
    Y = np.array(
        [[0.4, -0.4, -0.9], [1.0, -0.1, -0.4], [-0.0, -0.5, 1.1], [-1.8, -1.1, 0.3]]
    )
    np.testing.assert_allclose(
        approximate_covariance_matrix_from_ensembles(X, Y),
        np.array(
            [
                [-1.035, -1.15833333, 0.66, 1.56333333],
                [0.465, 0.36166667, -1.08, -1.09666667],
            ]
        ),
    )

    # Verify against numpy.cov

    np.testing.assert_allclose(
        np.cov(X, rowvar=True, ddof=1),
        approximate_covariance_matrix_from_ensembles(X, X),
    )


def test_get_ensemble_variance():
    ens = np.random.random((6, 6)) * 4.0

    np.testing.assert_almost_equal(
        approximate_covariance_matrix_from_ensembles(ens, ens).diagonal(),
        get_ensemble_variance(ens),
    )

    # must be 2D !
    with pytest.raises(ValueError):
        get_ensemble_variance(np.array((3.0)))


def test_inflate_ensemble_around_its_mean_factor_1() -> None:
    ensemble = np.ones((10, 10))
    inflated = inflate_ensemble_around_its_mean(ensemble, 1.0)

    np.testing.assert_equal(ensemble, inflated)


def test_inflate_ensemble_around_its_mean_random() -> None:
    rng = np.random.default_rng(0)
    ensemble = rng.normal(1.0, 2.0, size=(10, 10))
    inflated = inflate_ensemble_around_its_mean(ensemble, 2.0)

    # std * 2
    np.testing.assert_allclose(np.std(ensemble, axis=1) * 2.0, np.std(inflated, axis=1))
    # the mean does not change
    np.testing.assert_allclose(np.mean(ensemble, axis=1), np.mean(inflated, axis=1))


@pytest.mark.parametrize(
    "d_pred,assimilation_step,expected_exception",
    [
        (  # simple case
            np.ones((20, 30)),
            0,
            does_not_raise(),
        ),
        (  # 1D vector
            np.ones(20),
            0,
            does_not_raise(),
        ),
        (
            np.array(
                [[0.2, 0.2, 0.2, np.nan], [0.2, 0.2, 0.2, 0.2], [np.nan, 0.2, 0.2, 0.2]]
            ).T,
            0,
            pytest.raises(
                Exception,
                match=(
                    r"Something went wrong with the initial ensemble predictions  "
                    r"-> NaN values are found in predictions for members \[0, 2\] !"
                ),
            ),
        ),
        (
            np.array([[0.2, 0.2, 0.2, np.nan]]).T,
            0,
            pytest.raises(
                Exception,
                match=(
                    r"Something went wrong with the initial ensemble predictions  "
                    r"-> NaN values are found in predictions for members \[0\] !"
                ),
            ),
        ),
        (
            np.array(
                [[0.2, 0.2, 0.2, np.nan], [0.2, 0.2, 0.2, 0.2], [np.nan, 0.2, 0.2, 0.2]]
            ).T,
            2,
            pytest.raises(
                Exception,
                match=(
                    r"Something went wrong  after assimilation step 2 -> "
                    r"NaN values are found in predictions for members \[0, 2\] !"
                ),
            ),
        ),
    ],
)
def test_check_nans_in_predictions(d_pred, assimilation_step, expected_exception):
    with expected_exception:
        check_nans_in_predictions(d_pred, assimilation_step)
