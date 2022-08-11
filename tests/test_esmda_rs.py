"""
General test for the Ensemble-Smoother with Multiple Data Assimilation.

@author: acollet
"""
import numpy as np

from pyesmda import ESMDA_RS


def test_normalized_objective_function():

    pred = np.ones((20))
    obs = np.ones((20)) * 2.0
    obs_cov = np.diag(np.ones((20)) * 0.5)
    assert ESMDA_RS.compute_normalized_objective_function(pred, obs, obs_cov) == 1.0

    obs_cov = np.diag(np.ones((20)) * 2.0)
    assert ESMDA_RS.compute_normalized_objective_function(pred, obs, obs_cov) == 0.25


def test_ensemble_average_normalized_objective_function():

    pred = np.ones((10, 20))
    obs = np.ones((20)) * 2
    obs_cov = np.diag(np.ones((20)) * 0.5)
    assert (
        ESMDA_RS.compute_ensemble_average_normalized_objective_function(
            pred, obs, obs_cov
        )
        == 1.0
    )

    obs_cov = np.diag(np.ones((20)) * 2.0)
    assert (
        ESMDA_RS.compute_ensemble_average_normalized_objective_function(
            pred, obs, obs_cov
        )
        == 0.25
    )


def exponential(p, x):
    """
    Simple exponential function with an amplitude and change factor.

    Parameters
    ----------
    p : tuple, list
        Parameters vector: amplitude i.e. initial value and change factor.
    x : np.array
        Independent variable (e.g. time).

    Returns
    -------
    np.array
        Result.

    """
    return p[0] * np.exp(x * p[1])


def forward_model(m_ensemble, x):
    """
    Wrap the non-linear observation model (forward model).

    Function calling the non-linear observation model (forward model).
    for all ensemble members and returning the predicted data for
    each ensemble member.

    Parameters
    ----------
    m_ensemble : np.array
        Initial ensemble of N_{e} parameters vector..
    x : np.array
        Independent variable (e.g. time).

    Returns
    -------
    d_pred: np.array
        Predicted data for each ensemble member.
    """
    # Initiate an array of predicted results.
    d_pred = np.zeros([m_ensemble.shape[0], x.shape[0]])
    for j in range(m_ensemble.shape[0]):
        # Calling the forward model for each member of the ensemble
        d_pred[j, :] = exponential(m_ensemble[j, :], x)
    return d_pred


def test_esmda_rs_exponential_case():
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
    m_ensemble = np.stack((ma, mb), axis=1)

    # Observation error covariance matrix
    cov_obs = np.diag([1.0] * obs.shape[0])

    # A priori estimated parameters covariance
    cov_m_prior = np.array([[30, 0.0], [0.0, 0.01]])

    # Bounds on parameters (size m * 2)
    m_bounds = np.array([[0.0, 50.0], [-1.0, 1.0]])
    m_bounds = None

    # This is just for the test
    # cov_mm_inflation_factors: list[float] = [1.2]

    solver = ESMDA_RS(
        obs,
        m_ensemble,
        cov_obs,
        cov_m_prior,
        forward_model,
        forward_model_args=(x,),
        # cov_mm_inflation_factors=cov_mm_inflation_factors,
        m_bounds=m_bounds,
        md_correlation_matrix=np.ones((m_ensemble.shape[1], obs.size)),
        dd_correlation_matrix=np.ones((obs.size, obs.size)),
        save_ensembles_history=True,
        seed=0,
    )
    # Call the ES-MDA solver
    solver.solve()

    # Assert that the parameters are found with a 5% accuracy.
    assert np.isclose(
        np.average(solver.m_prior, axis=0), np.array([a, b]), rtol=5e-2
    ).all()

    # Get the uncertainty on the parameters
    a_std, b_std = np.sqrt(np.diagonal(solver.cov_mm))

    assert np.isclose(
        np.array([a_std, b_std]), np.array([1.23e-1, 7.49e-5]), rtol=5e-2
    ).all()

    assert solver.n_assimilations == 4
