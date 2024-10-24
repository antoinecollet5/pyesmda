"""
General test for the Ensemble-Smoother with Multiple Data Assimilation.

@author: acollet
"""

from contextlib import contextmanager

import numpy as np
import pytest
from pyesmda import ESMDA, ESMDAInversionType
from pyesmda.localization import FixedLocalization
from pyesmda.utils import NDArrayFloat


@contextmanager
def does_not_raise():
    yield


def empty_forward_model(*args, **kwargs) -> None:
    return


@pytest.mark.parametrize(
    "args,kwargs,expected_exception",
    [
        (  # simple construction
            (
                np.zeros(20),
                np.zeros((10, 8)),
                np.diag(np.ones(20)),
                empty_forward_model,
            ),
            {},
            does_not_raise(),
        ),
        (  # issue with stdev_d
            (
                np.zeros(20),
                np.zeros((10, 8)),
                np.diag(np.ones((8))),
                empty_forward_model,
            ),
            {},
            pytest.raises(
                ValueError,
                match=(r"cov_obs must be a 2D matrix with " r"dimensions \(20, 20\)."),
            ),
        ),
        (  # issue with stdev_d
            (np.zeros(20), np.zeros((10, 8)), np.ones((9,)), empty_forward_model),
            {},
            pytest.raises(
                ValueError,
                match=(r"cov_obs must be a 2D matrix with " r"dimensions \(20, 20\)."),
            ),
        ),
        (  # issue with stdev_d
            (np.zeros(20), np.zeros((10, 8)), np.ones((20, 19)), empty_forward_model),
            {},
            pytest.raises(
                ValueError,
                match=(r"cov_obs must be a 2D matrix with " r"dimensions \(20, 20\)."),
            ),
        ),
        (  # issue with stdev_d
            (
                np.zeros(20),
                np.zeros((10, 8)),
                np.ones((20, 20, 20)),
                empty_forward_model,
            ),
            {},
            pytest.raises(
                ValueError,
                match=(r"cov_obs must be a 2D matrix with " r"dimensions \(20, 20\)."),
            ),
        ),
        (  # normal working with n_assimilations
            (np.zeros(20), np.zeros((10, 8)), np.ones((20)), empty_forward_model),
            {
                "n_assimilations": 4,
            },
            does_not_raise(),
        ),
        (
            (np.zeros(20), np.zeros((10, 8)), np.ones((20)), empty_forward_model),
            {
                "n_assimilations": 4.5,  # Not a valid number of assimilations
            },
            pytest.raises(
                TypeError,
                match="The number of assimilations must be a positive integer.",
            ),
        ),
        (
            (
                np.zeros(20),
                np.zeros((10, 8)),
                np.diag(np.ones((20, 20))),
                empty_forward_model,
            ),
            {
                "forward_model_args": (22, 45, "some_arg"),
                "forward_model_kwargs": {"some_kwargs": "str", "some_other": 98.0},
            },
            does_not_raise(),
        ),
        (
            (
                np.zeros(20),
                np.zeros((10, 8)),
                np.diag(np.ones((20, 20))),
                empty_forward_model,
            ),
            {
                "n_assimilations": -1,  # Not a valid number of assimilations
            },
            pytest.raises(
                ValueError, match=r"The number of assimilations must be 1 or more."
            ),
        ),
        (
            (
                np.zeros(20),
                np.zeros((10, 8)),
                np.diag(np.ones((20, 20))),
                empty_forward_model,
            ),
            {
                "m_bounds": np.zeros((10, 2)),
            },
            does_not_raise(),
        ),
        (
            (
                np.zeros(20),
                np.zeros((10, 8)),
                np.diag(np.ones((20, 20))),
                empty_forward_model,
            ),
            {
                "m_bounds": np.zeros((3, 7)),
            },
            pytest.raises(
                ValueError,
                match=(
                    r"m_bounds is of shape \(3, 7\) while"
                    r" it should be of shape \(10, 2\)"
                ),
            ),
        ),
        (
            (
                np.zeros(20),
                np.zeros((10, 8)),
                np.diag(np.ones((20, 20))),
                empty_forward_model,
            ),
            {
                "n_assimilations": 3,
                "cov_obs_inflation_factors": np.ones(5),
            },
            pytest.raises(
                ValueError,
                match=(
                    r"The length of cov_obs_inflation_factors "
                    "should match n_assimilations"
                ),
            ),
        ),
        (
            (
                np.zeros(20),
                np.zeros((10, 8)),
                np.diag(np.ones((20, 20))),
                empty_forward_model,
            ),
            {
                "C_DD_localization": FixedLocalization(np.zeros((20, 20))),
            },
            does_not_raise(),
        ),
        (
            (
                np.zeros(20),
                np.zeros((10, 8)),
                np.diag(np.ones((20, 20))),
                empty_forward_model,
            ),
            {
                "C_MD_localization": FixedLocalization(np.zeros((10, 20))),
            },
            does_not_raise(),
        ),
        (
            (
                np.zeros(20),
                np.zeros((10, 8)),
                np.diag(np.ones((20, 20))),
                empty_forward_model,
            ),
            {
                "C_DD_localization": FixedLocalization(np.zeros((20, 20))),
            },
            does_not_raise(),
        ),
        (
            (
                np.zeros(20),
                np.zeros((10, 8)),
                np.diag(np.ones((20, 20))),
                empty_forward_model,
            ),
            {
                "C_MD_localization": FixedLocalization(np.zeros((10, 20))),
            },
            does_not_raise(),
        ),
        (
            (
                np.zeros(20),
                np.zeros((10, 8)),
                np.diag(np.ones((20, 20))),
                empty_forward_model,
            ),
            {
                "C_DD_localization": FixedLocalization(np.zeros((20))),
            },
            pytest.raises(
                ValueError,
                match=(
                    r"C_DD_localization must be a 2D "
                    r"matrix with dimensions \(20, 20\)."
                ),
            ),
        ),
        (
            (
                np.zeros(20),
                np.zeros((10, 8)),
                np.diag(np.ones((20, 20))),
                empty_forward_model,
            ),
            {
                "C_MD_localization": FixedLocalization(np.zeros((10))),
            },
            pytest.raises(
                ValueError,
                match=(
                    r"C_MD_localization must be a 2D "
                    r"matrix with dimensions \(10, 20\)."
                ),
            ),
        ),
        (
            (
                np.zeros(20),
                np.zeros((10, 8)),
                np.diag(np.ones((20, 20))),
                empty_forward_model,
            ),
            {
                "C_DD_localization": FixedLocalization(np.zeros((20, 19))),
            },
            pytest.raises(
                ValueError,
                match=(
                    r"C_DD_localization must be a 2D "
                    r"matrix with dimensions \(20, 20\)."
                ),
            ),
        ),
        (
            (
                np.zeros(20),
                np.zeros((10, 8)),
                np.diag(np.ones((20, 20))),
                empty_forward_model,
            ),
            {
                "C_MD_localization": FixedLocalization(np.zeros((10, 10))),
            },
            pytest.raises(
                ValueError,
                match=(
                    r"C_MD_localization must be a 2D "
                    r"matrix with dimensions \(10, 20\)."
                ),
            ),
        ),
        (
            (
                np.zeros(20),
                np.zeros((10, 8)),
                np.diag(np.ones((20, 20))),
                empty_forward_model,
            ),
            {
                "inversion_type": "smth_not_valid",
            },
            pytest.raises(
                ValueError,
                match=(
                    r"smth_not_valid is not a supported inversion type! "
                    r"Choose among \['naive', 'exact_cholesky', 'exact_lstq', "
                    r"'exact_woodbury', 'rescaled', 'subspace', "
                    r"'subspace_rescaled'\]"
                ),
            ),
        ),
        (
            (
                np.zeros(20),
                np.zeros((10, 8)),
                np.diag(np.ones((20, 20))),
                empty_forward_model,
            ),
            {
                "truncation": 1.1,
            },
            pytest.raises(
                ValueError,
                match="The truncation number should be in ]0, 1]!",
            ),
        ),
    ],
)
def test_constructor(args, kwargs, expected_exception) -> ESMDA:
    with expected_exception:
        esmda = ESMDA(*args, **kwargs)

        if "cov_obs_inflation_factors " not in kwargs.keys():
            _sum = 0
            for val in esmda.cov_obs_inflation_factors:
                _sum += 1 / val
                assert val == esmda.n_assimilations
            assert _sum == 1.0


def exponential(p, x) -> NDArrayFloat:
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
    d_pred = np.zeros([x.shape[0], m_ensemble.shape[1]])
    for j in range(m_ensemble.shape[1]):
        # Calling the forward model for each member of the ensemble
        d_pred[:, j] = exponential(m_ensemble[:, j], x)
    return d_pred


expected_std = {
    ESMDAInversionType.NAIVE: [1.05e-1, 6.31e-5],
    ESMDAInversionType.EXACT_CHOLESKY: [1.05e-1, 6.31e-5],
    ESMDAInversionType.EXACT_LSTSQ: [1.055e-1, 6.31e-5],
    ESMDAInversionType.EXACT_WOODBURY: [1.055e-1, 6.31e-5],
    ESMDAInversionType.RESCALED: [1.05e-1, 6.64e-5],
    ESMDAInversionType.SUBSPACE: [1.05e-01, 6.31e-05],
    ESMDAInversionType.SUBSPACE_RESCALED: [0.105, 6.30e-05],
}


@pytest.mark.parametrize("is_cov_obs_2D", [True, False])
@pytest.mark.parametrize(
    "inversion_type",
    [
        ESMDAInversionType.NAIVE,
        ESMDAInversionType.EXACT_CHOLESKY,
        ESMDAInversionType.EXACT_LSTSQ,
        ESMDAInversionType.EXACT_WOODBURY,
        ESMDAInversionType.RESCALED,
        ESMDAInversionType.SUBSPACE,
        ESMDAInversionType.SUBSPACE_RESCALED,
    ],
)
def test_esmda_exponential_case(
    is_cov_obs_2D: bool, inversion_type: ESMDAInversionType
) -> None:
    """Test the ES-MDA on a simple synthetic case with two parameters."""
    seed = 0
    rng = np.random.default_rng(seed=seed)

    a = 10.0
    b = -0.0020
    # timesteps
    x = np.arange(500)
    obs = exponential((a, b), x) + rng.normal(0.0, 1.0, 500)
    # Initiate an ensemble of (a, b) parameters
    n_ensemble = 100
    # Uniform law for the parameter a ensemble
    ma = rng.uniform(low=-10.0, high=50.0, size=n_ensemble)
    # Uniform law for the parameter b ensemble
    mb = rng.uniform(low=-0.003, high=0.001, size=n_ensemble)
    # Prior ensemble
    m_ensemble = np.stack((ma, mb), axis=0)

    # Observation error covariance matrix
    cov_obs = np.ones(obs.size, dtype=np.float64)
    if is_cov_obs_2D:
        cov_obs = np.diag(cov_obs)

    # Bounds on parameters (size m * 2)
    m_bounds = np.array([[0.0, 50.0], [-1.0, 1.0]])

    # Number of assimilations
    n_assimilations = 3

    # Use a geometric suite (see procedure un evensen 2018) to compte alphas.
    # Also explained in Torrado 2021 (see her PhD manuscript.)
    cov_obs_inflation_geo = 1.2
    cov_obs_inflation_factors: list[float] = [1.1]
    for assimilation in range(1, n_assimilations):
        cov_obs_inflation_factors.append(
            cov_obs_inflation_factors[assimilation - 1] / cov_obs_inflation_geo
        )
    scaling_factor: float = np.sum(1 / np.array(cov_obs_inflation_factors))
    cov_obs_inflation_factors = [
        alpha * scaling_factor for alpha in cov_obs_inflation_factors
    ]

    np.testing.assert_almost_equal(sum(1.0 / np.array(cov_obs_inflation_factors)), 1.0)

    solver = ESMDA(
        obs,
        m_ensemble,
        cov_obs,
        forward_model,
        forward_model_args=(x,),
        forward_model_kwargs={},
        n_assimilations=n_assimilations,
        cov_obs_inflation_factors=cov_obs_inflation_factors,
        cov_mm_inflation_factor=1.2,
        m_bounds=m_bounds,
        save_ensembles_history=True,
        inversion_type=inversion_type,
        random_state=seed,
        truncation=0.99,
    )
    # Call the ES-MDA solver
    solver.solve()

    # Assert that the parameters are found with a 10% accuracy.
    assert np.isclose(
        np.average(solver.m_prior, axis=1), np.array([a, b]), rtol=5e-2
    ).all()

    # Get the uncertainty on the parameters
    a_std, b_std = np.sqrt(np.diagonal(solver.cov_mm))

    assert np.isclose(
        np.array([a_std, b_std]), np.array(expected_std[inversion_type]), rtol=0.5
    ).all()

    np.testing.assert_almost_equal(
        np.sum(1 / np.array(solver.cov_obs_inflation_factors)), 1.0
    )


@pytest.mark.parametrize(
    "batch_size, cov_obs_dim, is_parallel_analyse_step, expected_n_batches",
    [
        (1, 1, False, 2),
        (2, 2, False, 1),
        (2, 1, True, 1),
        (1, 1, True, 2),
    ],
)
def test_esmda_exponential_case_batch(
    batch_size, cov_obs_dim, is_parallel_analyse_step, expected_n_batches
) -> None:
    """Test the ES-MDA on a simple synthetic case with two parameters."""
    seed = 0
    rng = np.random.default_rng(seed=seed)

    a = 10.0
    b = -0.0020

    # timesteps
    x = np.arange(500)
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
    cov_obs = np.ones(obs.shape[0])
    if cov_obs_dim == 2:
        cov_obs = np.diag(cov_obs)

    # Bounds on parameters (size m * 2)
    m_bounds = np.array([[0.0, 50.0], [-1.0, 1.0]])

    # Number of assimilations
    n_assimilations = 3

    # Use a geometric suite (see procedure un evensen 2018) to compte alphas.
    # Also explained in Torrado 2021 (see her PhD manuscript.)
    cov_obs_inflation_geo = 1.2
    cov_obs_inflation_factors: list[float] = [1.1]
    for step in range(1, n_assimilations):
        cov_obs_inflation_factors.append(
            cov_obs_inflation_factors[step - 1] / cov_obs_inflation_geo
        )
    scaling_factor: float = np.sum(1 / np.array(cov_obs_inflation_factors))
    cov_obs_inflation_factors = [
        alpha * scaling_factor for alpha in cov_obs_inflation_factors
    ]

    np.testing.assert_almost_equal(sum(1.0 / np.array(cov_obs_inflation_factors)), 1.0)

    solver = ESMDA(
        obs,
        m_ensemble,
        cov_obs,
        forward_model,
        forward_model_args=(x,),
        forward_model_kwargs={},
        n_assimilations=n_assimilations,
        cov_obs_inflation_factors=cov_obs_inflation_factors,
        cov_mm_inflation_factor=1.2,
        C_MD_localization=FixedLocalization(np.ones((m_ensemble.shape[0], obs.size))),
        C_DD_localization=FixedLocalization(np.ones((obs.size, obs.size))),
        m_bounds=m_bounds,
        save_ensembles_history=True,
        random_state=seed,
        batch_size=batch_size,
        is_parallel_analyse_step=is_parallel_analyse_step,
    )

    assert solver.n_batches == expected_n_batches

    # Call the ES-MDA solver
    solver.solve()

    # Assert that the parameters are found with a 5% accuracy.
    assert np.isclose(
        np.average(solver.m_prior, axis=1), np.array([a, b]), rtol=5e-2
    ).all()

    # Get the approximated parameters
    a_approx, b_approx = np.average(solver.m_prior, axis=1)

    # Get the uncertainty on the parameters
    a_std, b_std = np.sqrt(np.diagonal(solver.cov_mm))

    print(f"a = {a_approx:.5f} +/- {a_std:.4E}")
    print(f"b = {b_approx:.5f} +/- {b_std: 4E}")

    # Assert that the parameters are found with a 5% accuracy.
    assert np.isclose(
        np.average(solver.m_prior, axis=1), np.array([a, b]), rtol=5e-2
    ).all()

    # Get the uncertainty on the parameters
    a_std, b_std = np.sqrt(np.diagonal(solver.cov_mm))

    assert np.isclose(
        np.array([a_std, b_std]), np.array([1.01266593e-01, 5.73889089e-05]), rtol=5e-2
    ).all()

    # The sum of the inverse of inflation factors should be 1.0
    np.testing.assert_almost_equal(
        np.sum(1 / np.array(solver.cov_obs_inflation_factors)), 1.0
    )
