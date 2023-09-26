import numpy as np
import scipy as sp

from pyesmda.inversion import ESMDAInversionType, get_localized_cmd_multi_dot


def test_inversion_type() -> None:
    for t in ESMDAInversionType.to_list():
        assert t.value == ESMDAInversionType(t.value)

    assert not "test" == ESMDAInversionType.NAIVE
    assert "test" != ESMDAInversionType.NAIVE
    assert not 2.0 == ESMDAInversionType.NAIVE


def test_get_localized_cmd_multi_dot() -> None:
    n_ensemble = 50
    n_x = 5000
    n_obs = 500

    X = np.random.random((n_x, n_ensemble))
    Y = np.random.random((n_obs, n_ensemble))

    # Test 1: no other input
    get_localized_cmd_multi_dot(X, Y)

    # Test 2: with args
    K = np.random.random((n_obs, n_x))
    get_localized_cmd_multi_dot(X, Y, K)

    # Test 3 with multiple args
    get_localized_cmd_multi_dot(X, Y, Y, X.T)

    # Test 4 with localization
    cov = sp.sparse.csr_array(np.ones((n_x, n_obs)))

    np.testing.assert_allclose(
        get_localized_cmd_multi_dot(X, Y, Y, X.T, md_corr_mat=cov),
        get_localized_cmd_multi_dot(X, Y, Y, X.T),
        rtol=1e-6,
        atol=1e-6,
    )
