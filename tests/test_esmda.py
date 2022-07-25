"""
General test for the Ensemble-Smoother with Multiple Data Assimilation.

@author: acollet
"""
from contextlib import contextmanager

import numpy as np
import pytest

from pyesmda import ESMDA


@contextmanager
def does_not_raise():
    yield


def empty_forward_model() -> None:
    return


@pytest.mark.parametrize(
    "args,kwargs,expected_exception",
    [
        (  # simple construction
            (np.zeros(10), np.zeros((10, 10)), np.zeros((10, 10)), empty_forward_model),
            {},
            does_not_raise(),
        ),
        (  # issue with stdev_d
            (np.zeros(10), np.zeros((10, 10)), np.zeros((10)), empty_forward_model),
            {},
            pytest.raises(
                ValueError,
                match=(
                    r"cov_d must be a square matrix with "
                    r"same dimensions as the observations vector."
                ),
            ),
        ),
        (  # issue with stdev_d
            (np.zeros(10), np.zeros((10, 10)), np.zeros((9, 9)), empty_forward_model),
            {},
            pytest.raises(
                ValueError,
                match=(
                    r"cov_d must be a square matrix with same dimensions "
                    r"as the observations vector."
                ),
            ),
        ),
        (  # normal working with n_assimilations
            (np.zeros(10), np.zeros((10, 10)), np.zeros((10, 10)), empty_forward_model),
            {
                "n_assimilations": 4,
            },
            does_not_raise(),
        ),
        (
            (np.zeros(10), np.zeros((10, 10)), np.zeros((10, 10)), empty_forward_model),
            {
                "n_assimilations": 4.5,  # Not a valid number of assimilations
            },
            pytest.raises(
                TypeError,
                match="The number of assimilations must be a positive integer.",
            ),
        ),
        (
            (np.zeros(10), np.zeros((10, 10)), np.zeros((10, 10)), empty_forward_model),
            {
                "forward_model_args": (22, 45, "some_arg"),
                "forward_model_kwargs": {"some_kwargs": "str", "some_other": 98.0},
            },
            does_not_raise(),
        ),
        (
            (np.zeros(10), np.zeros((10, 10)), np.zeros((10, 10)), empty_forward_model),
            {
                "n_assimilations": -1,  # Not a valid number of assimilations
            },
            pytest.raises(
                ValueError, match=r"The number of assimilations must be 1 or more."
            ),
        ),
        (
            (np.zeros(10), np.zeros((10, 10)), np.zeros((10, 10)), empty_forward_model),
            {
                "m_bounds": np.zeros((10, 2)),
            },
            does_not_raise(),
        ),
        (
            (np.zeros(10), np.zeros((10, 10)), np.zeros((10, 10)), empty_forward_model),
            {
                "m_bounds": np.zeros((3, 7)),
            },
            pytest.raises(
                ValueError,
                match=(
                    r"m_bounds is of size \(3, 7\) while"
                    r" it should be of size \(10, 2\)"
                ),
            ),
        ),
        (
            (np.zeros(10), np.zeros((10, 10)), np.zeros((10, 10)), empty_forward_model),
            {
                "n_assimilations": 3,
                "cov_d_inflation_factors": np.ones(5),
            },
            pytest.raises(
                ValueError,
                match=(
                    r"The length of cov_d_inflation_factors "
                    "should match n_assimilations"
                ),
            ),
        ),
    ],
)
def test_constructor(args, kwargs, expected_exception) -> ESMDA:

    with expected_exception:
        esmda = ESMDA(*args, **kwargs)

        if "cov_d_inflation_factors " not in kwargs.keys():
            for val in esmda.cov_d_inflation_factors:
                assert val == 1 / esmda.n_assimilations

        if "cov_mm_inflation_factors " not in kwargs.keys():
            for val in esmda.cov_mm_inflation_factors:
                assert val == 1.0


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
    """Test the ES-MDA on a simple synthetic case with two parameters."""
    a = 10.0
    b = -0.0020
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
    # Prior ensemble
    m_ensemble = np.stack((ma, mb), axis=1)

    # Observation error covariance matrix
    cov_d = np.diag([1.0] * obs.shape[0])

    # Bounds on parameters (size m * 2)
    m_bounds = np.array([[0.0, 50.0], [-1.0, 1.0]])
    m_bounds = None

    # Number of assimilations
    n_assimilations = 4

    # Use a geometric suite (see procedure un evensen 2018) to compte alphas.
    # Also explained in Torrado 2021 (see her PhD manuscript.)
    cov_d_inflation_geo = 1.2
    cov_d_inflation_factors: list[float] = [1.1]
    for a_step in range(1, n_assimilations):
        cov_d_inflation_factors.append(
            cov_d_inflation_factors[a_step - 1] / cov_d_inflation_geo
        )
    scaling_factor: float = np.sum(1 / np.array(cov_d_inflation_factors))
    cov_d_inflation_factors = [
        alpha * scaling_factor for alpha in cov_d_inflation_factors
    ]

    np.testing.assert_almost_equal(sum(1.0 / np.array(cov_d_inflation_factors)), 1.0)

    # This is just for the test
    cov_mm_inflation_factors: list[float] = [1.2] * n_assimilations

    solver = ESMDA(
        obs,
        m_ensemble,
        cov_d,
        forward_model,
        forward_model_args=(x,),
        forward_model_kwargs={},
        n_assimilations=n_assimilations,
        cov_d_inflation_factors=cov_d_inflation_factors,
        cov_mm_inflation_factors=cov_mm_inflation_factors,
        m_bounds=m_bounds,
        save_ensembles_history=True,
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
        np.array([a_std, b_std]), np.array([1.4e-1, 7.3e-5]), rtol=5e-2
    ).all()
