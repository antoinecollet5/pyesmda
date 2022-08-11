"""
General test for the Ensemble-Smoother with Multiple Data Assimilation.

@author: acollet
"""
import numpy as np

from pyesmda import ESMDA_RS


def test_normalized_objective_function():

    pred = np.ones((20))
    obs = np.ones((20)) * 2.0
    obs_cov = np.diagonal(np.ones((20, 20))) * 0.5
    assert ESMDA_RS.compute_normalized_objective_function(pred, obs, obs_cov) == 1.0

    obs_cov = np.diagonal(np.ones((20, 20))) * 2.0
    assert ESMDA_RS.compute_normalized_objective_function(pred, obs, obs_cov) == 0.25


def test_ensemble_average_normalized_objective_function():

    pred = np.ones((10, 20))
    obs = np.ones((20)) * 2
    obs_cov = np.diagonal(np.ones((20, 20))) * 0.5
    assert (
        ESMDA_RS.compute_ensemble_average_normalized_objective_function(
            pred, obs, obs_cov
        )
        == 1.0
    )

    obs_cov = np.diagonal(np.ones((20, 20))) * 2.0
    assert (
        ESMDA_RS.compute_ensemble_average_normalized_objective_function(
            pred, obs, obs_cov
        )
        == 0.25
    )
