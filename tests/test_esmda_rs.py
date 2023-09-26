"""
General test for the Ensemble-Smoother with Multiple Data Assimilation.

@author: acollet
"""
from typing import List

import numpy as np
import pytest

from pyesmda import ESMDA_RS

from .test_esmda import exponential, forward_model


@pytest.mark.parametrize(
    "is_use_std_m_prior,expected_uncertainties,expected_n_assimilations",
    [(True, [1.1e-1, 6.69e-5], 4), (False, [9.57e-2, 5.10e-5], 8)],
)
def test_esmda_rs_exponential_case(
    is_use_std_m_prior: bool,
    expected_uncertainties: List[float],
    expected_n_assimilations: int,
) -> None:
    """Test the ES-MDA on a simple synthetic case with two parameters."""
    a = 10.0
    b = -0.0020
    # timesteps
    x = np.arange(500)
    # Noisy signal with predictable noise
    rng = np.random.default_rng(0)
    obs = exponential((a, b), x) + rng.normal(0.0, 1.0, 500)
    # Initiate an ensemble of (a, b) parameters
    n_ensemble = 100  # size of the ensemble
    # Uniform law for the parameter a ensemble
    ma = rng.uniform(low=-10.0, high=50.0, size=n_ensemble)
    # Uniform law for the parameter b ensemble
    mb = rng.uniform(low=-0.001, high=0.01, size=n_ensemble)
    # Prior ensemble
    m_ensemble = np.stack((ma, mb), axis=0)

    # Observation error covariance matrix
    cov_obs = np.diag([1.0] * obs.shape[0])

    # A priori estimated parameters standard deviation
    if is_use_std_m_prior:
        std_m_prior = np.array([30, 0.01])
    else:
        std_m_prior = None

    # Bounds on parameters (size m * 2)
    m_bounds = np.array([[0.0, 50.0], [-1.0, 1.0]])
    m_bounds = None

    # This is just for the test
    cov_mm_inflation_factor: float = 0.9

    solver = ESMDA_RS(
        obs,
        m_ensemble,
        cov_obs,
        forward_model,
        forward_model_args=(x,),
        cov_mm_inflation_factor=cov_mm_inflation_factor,
        m_bounds=m_bounds,
        md_correlation_matrix=np.ones((m_ensemble.shape[0], obs.size)),
        dd_correlation_matrix=np.ones((obs.size, obs.size)),
        save_ensembles_history=True,
        std_m_prior=std_m_prior,
        seed=123,  # deprecated and not used anymore
        random_state=123,
    )
    # Call the ES-MDA solver
    solver.solve()

    # Assert that the parameters are found with a 5% accuracy.
    assert np.isclose(
        np.average(solver.m_prior, axis=1), np.array([a, b]), rtol=1e-1
    ).all()

    # Get the uncertainty on the parameters
    a_std, b_std = np.sqrt(np.diagonal(solver.cov_mm))

    assert np.isclose(
        np.array([a_std, b_std]), np.array(expected_uncertainties), rtol=1e-1
    ).all()

    assert solver.n_assimilations == expected_n_assimilations


@pytest.mark.parametrize(
    "batch_size, is_parallel_analyse_step, "
    "expected_uncertainties, expected_n_assimilations",
    [
        (1, False, [1.1e-1, 6.69e-5], 4),
        (2, False, [1.1e-1, 6.69e-5], 4),
        (2, True, [1.1e-1, 6.69e-5], 4),
    ],
)
def test_esmda_exponential_case_batch(
    batch_size,
    is_parallel_analyse_step,
    expected_uncertainties,
    expected_n_assimilations,
) -> None:
    """Test the ES-MDA on a simple synthetic case with two parameters."""
    a = 10.0
    b = -0.0020
    # timesteps
    x = np.arange(500)
    # Noisy signal with predictable noise
    rng = np.random.default_rng(0)
    obs = exponential((a, b), x) + rng.normal(0.0, 1.0, 500)
    # Initiate an ensemble of (a, b) parameters
    n_ensemble = 100  # size of the ensemble
    # Uniform law for the parameter a ensemble
    ma = rng.uniform(low=-10.0, high=50.0, size=n_ensemble)
    # Uniform law for the parameter b ensemble
    mb = rng.uniform(low=-0.001, high=0.01, size=n_ensemble)
    # Prior ensemble
    m_ensemble = np.stack((ma, mb), axis=0)

    # Observation error covariance matrix
    cov_obs = np.diag([1.0] * obs.shape[0])

    std_m_prior = np.array([30, 0.01])

    # Bounds on parameters (size m * 2)
    m_bounds = np.array([[0.0, 50.0], [-1.0, 1.0]])
    m_bounds = None

    # This is just for the test
    cov_mm_inflation_factor: float = 0.9

    solver = ESMDA_RS(
        obs,
        m_ensemble,
        cov_obs,
        forward_model,
        forward_model_args=(x,),
        cov_mm_inflation_factor=cov_mm_inflation_factor,
        m_bounds=m_bounds,
        md_correlation_matrix=np.ones((m_ensemble.shape[0], obs.size)),
        dd_correlation_matrix=np.ones((obs.size, obs.size)),
        save_ensembles_history=True,
        std_m_prior=std_m_prior,
        seed=123,  # deprecated and not used anymore
        random_state=123,
        batch_size=batch_size,
        is_parallel_analyse_step=is_parallel_analyse_step,
    )
    # Call the ES-MDA solver
    solver.solve()

    # Assert that the parameters are found with a 5% accuracy.
    assert np.isclose(
        np.average(solver.m_prior, axis=1), np.array([a, b]), rtol=5e-2
    ).all()

    # Get the uncertainty on the parameters
    a_std, b_std = np.sqrt(np.diagonal(solver.cov_mm))

    assert np.isclose(
        np.array([a_std, b_std]), np.array(expected_uncertainties), rtol=5e-2
    ).all()

    assert solver.n_assimilations == expected_n_assimilations
