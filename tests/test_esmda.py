"""
General test for the Ensemble-Smoother with Mulitple Data Assimilation.

@author: acollet
"""
import numpy as np
from pyesmda import ESMDA


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


def test_esmda_exponential_case():
    """Test the ES-MDA on a simple syntetic case with two parameters."""
    a = 10.0
    b = - 0.0020
    # timesteps
    x = np.arange(500)
    # Noisy signal with predictable noise
    np.random.seed(0)
    obs = exponential((a, b), x) + np.random.normal(0.0, 1.0, 500)
    # Initiate an ensemble of (a, b) parameters
    n_ensemble = 100  # size of the ensemble
    # Uniform law for the parameter a ensemble
    ma = np.random.uniform(low=-10.0, high=50.0, size=n_ensemble)
    # Uniform law for the parameter b ensemble
    mb = np.random.uniform(low=-0.001, high=0.01, size=n_ensemble)
    m_ensemble = np.stack((ma, mb), axis=1)  # Prior ensemble

    # Observation error covariance matrix
    stdev_d = np.diag([1.0] * obs.shape[0])
    # Parameters error covariance matrix
    stdev_m = np.multiply([1.0, 0.01] * 1, 1.0)

    # Bounds on parameters (size m * 2)
    m_bounds = np.array([[0.0, 50.0], [-1.0, 1.0]])
    m_bounds = None

    alpha = np.zeros([4])
    alpha[0] = 9.33
    alpha[1] = 7
    alpha[2] = 4
    alpha[3] = 2

    # Number of assimilations
    n_assimilation = 4
    solveur = ESMDA(obs, m_ensemble, stdev_d, stdev_m,
                    forward_model, forward_model_args=(x,),
                    forward_model_kwargs={},
                    n_assimilation=n_assimilation,
                    alpha=alpha, m_bounds=m_bounds)
    # Call the ES-MDA solver
    solveur.solve()

    # Assert that the parameters are found with a 5% accuracy.
    assert np.isclose(solveur.m_mean[-1], np.array([a, b]), rtol=5e-2).all()
