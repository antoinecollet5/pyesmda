"""
Test for the Ensemble-Smoother with Multiple Data Assimilation utils.

@author: acollet
"""
from contextlib import contextmanager

import numpy as np
import pytest

from pyesmda.utils import (
    approximate_covariance_matrix_from_ensembles,
    check_nans_in_predictions,
    compute_ensemble_average_normalized_objective_function,
    compute_normalized_objective_function,
    inflate_ensemble_around_its_mean,
)


@contextmanager
def does_not_raise():
    yield


def test_normalized_objective_function():

    pred = np.ones((20))
    obs = np.ones((20)) * 2.0
    obs_cov = np.diag(np.ones((20)) * 0.5)
    assert compute_normalized_objective_function(pred, obs, obs_cov) == 1.0

    obs_cov = np.diag(np.ones((20)) * 2.0)
    assert compute_normalized_objective_function(pred, obs, obs_cov) == 0.25


def test_ensemble_average_normalized_objective_function():

    pred = np.ones((10, 20))
    obs = np.ones((20)) * 2
    obs_cov = np.diag(np.ones((20)) * 0.5)
    assert (
        compute_ensemble_average_normalized_objective_function(pred, obs, obs_cov)
        == 1.0
    )

    obs_cov = np.diag(np.ones((20)) * 2.0)
    assert (
        compute_ensemble_average_normalized_objective_function(pred, obs, obs_cov)
        == 0.25
    )


@pytest.mark.parametrize(
    "ens1,ens2,expected,expected_exception",
    [
        (  # simple construction
            np.ones((20, 30)),
            np.ones((20, 10)),
            np.zeros((30, 10)),
            does_not_raise(),
        ),
        (  # issue with stdev_d
            np.ones((10, 30)),
            np.ones((21, 10)),
            None,
            pytest.raises(
                ValueError,
                match=(
                    r"The ensemble should be 2D matrices with equal first dimension!"
                ),
            ),
        ),
        (  # issue with stdev_d
            np.ones((10)),
            np.ones((10, 10)),
            None,
            pytest.raises(
                ValueError,
                match=(
                    r"The ensemble should be 2D matrices with equal first dimension!"
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


def test_inflate_ensemble_around_its_mean_factor_1() -> None:

    ensemble = np.ones((10, 10))
    inflated = inflate_ensemble_around_its_mean(ensemble, 1.0)

    np.testing.assert_equal(ensemble, inflated)


def test_inflate_ensemble_around_its_mean_random() -> None:

    rng = np.random.default_rng(0)
    ensemble = rng.normal(1.0, 2.0, size=(10, 10))
    inflated = inflate_ensemble_around_its_mean(ensemble, 2.0)

    # std * 2
    np.testing.assert_allclose(np.std(ensemble, axis=0) * 2.0, np.std(inflated, axis=0))
    # the mean does not change
    np.testing.assert_allclose(np.mean(ensemble, axis=0), np.mean(inflated, axis=0))


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
            ),
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
            np.array([[0.2, 0.2, 0.2, np.nan]]),
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
            ),
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
