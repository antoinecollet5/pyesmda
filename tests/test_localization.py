from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from pyesmda.localization import (
    distances_to_weights_beta_cumulative,
    distances_to_weights_fifth_order,
)


@pytest.mark.parametrize(
    "beta,scaling_factor,expected_exception",
    [
        (
            -1.0,
            10.0,
            pytest.raises(
                ValueError, match=r"Beta \(-1.0\) should be positive or null !"
            ),
        ),
        (0.0, 10.0, does_not_raise()),
        (
            -1.0,
            -10.0,
            pytest.raises(
                ValueError,
                match=r"The scaling factor \(-10.0\) should be strictly positive !",
            ),
        ),
        (
            2.5,
            -10.0,
            pytest.raises(
                ValueError,
                match=r"The scaling factor \(-10.0\) should be strictly positive !",
            ),
        ),
        (
            2.2,
            0.0,
            pytest.raises(
                ValueError,
                match=r"The scaling factor \(0.0\) should be strictly positive !",
            ),
        ),
        (0.0, 10.0, does_not_raise()),
        (1.0, 20.0, does_not_raise()),
        (3.3, 26.7, does_not_raise()),
        (10.3, 26.7, does_not_raise()),
    ],
)
def test_distances_to_weights_beta_cumulative(
    beta, scaling_factor, expected_exception
) -> None:
    with expected_exception:
        res = distances_to_weights_beta_cumulative(
            np.linspace(0, 40.0, 100), beta=beta, scaling_factor=scaling_factor
        )
        assert res[0] == 1.0
        assert res[-1] == 0.0
        # Should be 0.5 at half the scaling factor
        assert (
            distances_to_weights_beta_cumulative(
                np.array([scaling_factor / 2]), beta=beta, scaling_factor=scaling_factor
            )
            == 0.5
        )


@pytest.mark.parametrize(
    "scaling_factor,expected_exception",
    [
        (
            -1.0,
            pytest.raises(
                ValueError,
                match=r"The scaling_factor \(-1.0\) should be strictly positive !",
            ),
        ),
        (
            0.0,
            pytest.raises(
                ValueError,
                match=r"The scaling_factor \(0.0\) should be strictly positive !",
            ),
        ),
        (0.5, does_not_raise()),
        (5.0, does_not_raise()),
        (50.0, does_not_raise()),
        (100.0, does_not_raise()),
    ],
)
def test_distances_to_weights_fifth_order(scaling_factor, expected_exception) -> None:
    with expected_exception:
        res = distances_to_weights_fifth_order(
            np.linspace(0, 300.0, 300), scaling_factor=scaling_factor
        )
        assert res[0] == 1.0
        assert res[-1] == 0.0
