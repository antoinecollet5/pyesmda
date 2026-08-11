# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021-2026 Antoine COLLET

from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
from pyesmda._localization import (
    CorrelationBasedLocalization,
    CorrelationTempering,
    CorrelationThresholding,
    CorrelationTransform,
    FixedLocalization,
    LocalizationStrategy,
    NoLocalization,
    _part1,
    _part2,
    _reversed_beta_cumulative,
    cov_to_corr,
    default_correlation_threshold,
    distances_to_weights_beta_cumulative,
    distances_to_weights_fifth_order,
    gc_correlation_tempering,
    gc_correlation_tempering_positive,
    make_correlation_threshold_callable,
)
from scipy.sparse import csr_matrix

# --------------------------------------------------------------------------- #
# Shared fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def ensembles(rng):
    """A pair of small, distinctly-shaped ensembles, plus a compatible
    correlation matrix, reused across several tests."""
    X = rng.random((3, 12))  # e.g. 3 parameters, 12 ensemble members
    Y = rng.random((4, 12))  # e.g. 4 observations, 12 ensemble members
    return X, Y


# --------------------------------------------------------------------------- #
# LocalizationStrategy (abstract base)
# --------------------------------------------------------------------------- #


class _MinimalStrategy(LocalizationStrategy):
    """Smallest possible concrete subclass, used to exercise the base
    class's own code: the concrete check_localization_shape no-op, and
    (via explicit super() calls) the abstract method bodies themselves."""

    def localize(self, X, Y, batch_slice=slice(None)):
        return super().localize(X, Y, batch_slice=batch_slice)

    def localize_multi_dot(self, X, Y, *args, batch_slice=slice(None)):
        return super().localize_multi_dot(X, Y, *args, batch_slice=batch_slice)


def test_localization_strategy_check_localization_shape_is_a_noop():
    strategy = _MinimalStrategy()
    # The base implementation ignores its arguments and does nothing; this
    # should not raise regardless of what's passed.
    assert strategy.check_localization_shape((5, 5), "whatever") is None
    assert strategy.check_localization_shape((1, 2, 3), "anything") is None


def test_localization_strategy_abstract_bodies_are_noop_via_super(ensembles):
    """The abstract localize()/localize_multi_dot() bodies are just `...`
    placeholders. A subclass would never normally call super() into them,
    but doing so here exercises those lines directly and confirms they're
    inert (return None) rather than doing something unexpected."""
    X, Y = ensembles
    strategy = _MinimalStrategy()
    assert strategy.localize(X, Y) is None
    assert strategy.localize_multi_dot(X, Y) is None


def test_localization_strategy_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        LocalizationStrategy()


# --------------------------------------------------------------------------- #
# FixedLocalization / NoLocalization
# --------------------------------------------------------------------------- #


def test_fixed_localization_init_without_matrix():
    loc = FixedLocalization()
    assert loc.correlation_matrix is None


def test_fixed_localization_init_with_dense_matrix():
    mat = np.eye(4)
    loc = FixedLocalization(mat)
    assert loc.correlation_matrix is not None
    assert loc.correlation_matrix.shape == (4, 4)


def test_fixed_localization_init_with_sparse_matrix():
    mat = csr_matrix(np.eye(4))
    loc = FixedLocalization(mat)
    assert loc.correlation_matrix.shape == (4, 4)


def test_fixed_localization_check_shape_noop_when_no_matrix():
    loc = FixedLocalization()
    # Should not raise, regardless of the (unused) expected shape.
    assert loc.check_localization_shape((10, 10), "C_DD") is None


def test_fixed_localization_check_shape_matching_ok():
    loc = FixedLocalization(np.eye(4))
    assert loc.check_localization_shape((4, 4), "C_DD") is None


def test_fixed_localization_check_shape_mismatch_raises():
    loc = FixedLocalization(np.eye(4))
    with pytest.raises(ValueError, match=r"C_DD must be a 2D matrix"):
        loc.check_localization_shape((5, 5), "C_DD")


def test_fixed_localization_localize_without_matrix(ensembles):
    X, Y = ensembles
    loc = FixedLocalization()
    result = loc.localize(X, Y)
    assert result.shape == (X.shape[0], Y.shape[0])


def test_fixed_localization_localize_with_matrix(ensembles):
    X, Y = ensembles
    corr = np.ones((X.shape[0], Y.shape[0]))
    loc = FixedLocalization(corr)
    result = loc.localize(X, Y)
    assert result.shape == (X.shape[0], Y.shape[0])
    # correlation_matrix is all ones, so this must equal the raw covariance
    ref = FixedLocalization().localize(X, Y)
    np.testing.assert_allclose(result, ref)


def test_fixed_localization_localize_with_matrix_and_batch_slice(rng):
    # batch_slice selects rows of the (full-size) correlation matrix to
    # match an X that has ALREADY been sliced to the same batch -- it does
    # not slice a full-size X after the fact.
    X_full = rng.random((5, 12))
    Y = rng.random((4, 12))
    corr_full = np.ones((5, 4))
    loc = FixedLocalization(corr_full)
    batch_slice = slice(1, 3)
    X_batch = X_full[batch_slice, :]
    result = loc.localize(X_batch, Y, batch_slice=batch_slice)
    assert result.shape == (2, Y.shape[0])


@pytest.mark.parametrize("has_matrix", [False, True])
@pytest.mark.parametrize("y_is_x", [False, True])
@pytest.mark.parametrize("with_args", [False, True])
def test_fixed_localization_localize_multi_dot_all_branches(
    ensembles, has_matrix, y_is_x, with_args
):
    X, Y = ensembles
    if y_is_x:
        Y = X  # exercise the `Y is X` aliasing branch
    corr = np.ones((X.shape[0], Y.shape[0])) if has_matrix else None
    loc = FixedLocalization(corr)

    args = (np.random.default_rng(1).random((Y.shape[0], 2)),) if with_args else ()
    result = loc.localize_multi_dot(X, Y, *args)

    if with_args:
        assert result.shape == (X.shape[0], 2)
    else:
        assert result.shape == (X.shape[0], Y.shape[0])


def test_no_localization_init_and_behavior(ensembles):
    X, Y = ensembles
    loc = NoLocalization()
    assert loc.correlation_matrix is None
    ref = FixedLocalization().localize(X, Y)
    np.testing.assert_allclose(loc.localize(X, Y), ref)


# --------------------------------------------------------------------------- #
# default_correlation_threshold
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "ne,expected",
    (((9, 1.0), (16, 0.75), (36, 0.5), (100, 0.3), (1000, 0.094868))),
)
def test_default_correlation_threshold(ne: int, expected: float) -> None:
    np.testing.assert_allclose(expected, default_correlation_threshold(ne), rtol=1e-5)


def test_default_correlation_threshold_zero() -> None:
    with pytest.raises(ValueError, match="The ensemble size cannot be zero!"):
        default_correlation_threshold(0)


def test_default_correlation_threshold_small_ensemble_clips_to_one() -> None:
    # 3 / sqrt(1) == 3, which must be clipped down to the max of 1.0.
    assert default_correlation_threshold(1) == 1.0


# --------------------------------------------------------------------------- #
# cov_to_corr
# --------------------------------------------------------------------------- #


def test_cov_to_corr_inplace_true_mutates_original_data():
    # Note: the function's final line reassigns through np.nan_to_num(),
    # which returns a new array by default -- so the *returned* object is
    # never literally `is cov`, even with inplace=True. What inplace=True
    # actually buys is skipping the initial .copy(): the original array's
    # underlying data is mutated by the /= steps either way, so check that
    # instead of object identity of the return value.
    cov = np.array([[4.0, 2.0], [2.0, 9.0]])
    original_id = id(cov)
    stds_x = np.array([2.0, 3.0])
    stds_y = np.array([2.0, 3.0])
    cov_to_corr(cov, stds_x, stds_y, inplace=True)
    assert id(cov) == original_id
    # The original array was mutated by the /= steps (no .copy() was made).
    np.testing.assert_allclose(cov, [[1.0, 1 / 3], [1 / 3, 1.0]])


def test_cov_to_corr_inplace_false_does_not_mutate_input():
    cov = np.array([[4.0, 2.0], [2.0, 9.0]])
    original = cov.copy()
    stds_x = np.array([2.0, 3.0])
    stds_y = np.array([2.0, 3.0])
    result = cov_to_corr(cov, stds_x, stds_y, inplace=False)
    np.testing.assert_allclose(cov, original)  # untouched
    assert result is not cov
    np.testing.assert_allclose(np.diag(result), [1.0, 1.0])


def test_cov_to_corr_handles_zero_std_via_nan_to_num():
    # A genuine 0/0 (both the covariance entry AND the std zero) is needed
    # to actually produce a `nan` -- dividing a *nonzero* value by a zero
    # std gives `inf`, not `nan`, and nan_to_num handles those separately
    # (see the inf/clipping test below).
    cov = np.array([[0.0, 2.0], [2.0, 9.0]])
    stds_x = np.array([0.0, 3.0])
    stds_y = np.array([2.0, 3.0])
    with pytest.warns(UserWarning):
        result = cov_to_corr(cov, stds_x, stds_y, inplace=False)
    assert not np.any(np.isnan(result))
    assert result[0, 0] == 0.0  # 0/0 -> nan -> replaced by 0.0


def test_cov_to_corr_nonzero_divided_by_zero_std_clips_via_inf():
    # Nonzero / 0 -> inf (not nan) -> nan_to_num's posinf replacement (a
    # huge finite value) -> clipped down to 1.0. Distinct code path from
    # the true 0/0 -> nan case above, even though both end up going through
    # nan_to_num and the final clip.
    cov = np.array([[4.0, 2.0], [2.0, 9.0]])
    stds_x = np.array([0.0, 3.0])
    stds_y = np.array([2.0, 3.0])
    with pytest.warns(UserWarning):
        result = cov_to_corr(cov, stds_x, stds_y, inplace=False)
    assert not np.any(np.isinf(result))
    assert result[0, 0] == 1.0
    assert result[0, 1] == 1.0


def test_cov_to_corr_warns_when_result_outside_valid_range():
    # Deliberately inconsistent inputs: cov entries too large relative to
    # the given stds for the result to be a valid correlation in [-1, 1].
    cov = np.array([[100.0]])
    stds_x = np.array([1.0])
    stds_y = np.array([1.0])
    with pytest.warns(UserWarning, match="Cross-correlation matrix has entries"):
        result = cov_to_corr(cov, stds_x, stds_y, inplace=False)
    # Still clipped into range despite the warning.
    assert result[0, 0] == 1.0


def test_cov_to_corr_clips_out_of_range_values():
    cov = np.array([[-100.0]])
    stds_x = np.array([1.0])
    stds_y = np.array([1.0])
    with pytest.warns(UserWarning):
        result = cov_to_corr(cov, stds_x, stds_y, inplace=False)
    assert result[0, 0] == -1.0


def test_cov_to_corr_no_warning_for_valid_correlation(recwarn):
    cov = np.array([[4.0, 2.0], [2.0, 9.0]])
    stds_x = np.array([2.0, 3.0])
    stds_y = np.array([2.0, 3.0])
    cov_to_corr(cov, stds_x, stds_y, inplace=False)
    assert len(recwarn) == 0


# --------------------------------------------------------------------------- #
# CorrelationTransform (abstract base)
# --------------------------------------------------------------------------- #


class _MinimalTransform(CorrelationTransform):
    def __call__(self, correlation_matrix, ne):
        return super().__call__(correlation_matrix, ne)


def test_correlation_transform_abstract_call_is_noop():
    t = _MinimalTransform()
    assert t(np.eye(2), 10) is None


def test_correlation_transform_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        CorrelationTransform()


# --------------------------------------------------------------------------- #
# make_correlation_threshold_callable
# --------------------------------------------------------------------------- #


def test_make_correlation_threshold_callable_none_returns_default():
    fn = make_correlation_threshold_callable(None)
    assert fn is default_correlation_threshold


def test_make_correlation_threshold_callable_passthrough_for_callable():
    def custom(ensemble_size):
        return 0.42

    fn = make_correlation_threshold_callable(custom)
    assert fn is custom
    assert fn(50) == 0.42


def test_make_correlation_threshold_callable_wraps_float_in_range():
    fn = make_correlation_threshold_callable(0.3)
    assert callable(fn)
    # Constant regardless of ensemble size -- covers the wrapper's own body.
    assert fn(5) == 0.3
    assert fn(500) == 0.3


@pytest.mark.parametrize("bad_value", [-0.1, 1.1, -5.0, 2.0])
def test_make_correlation_threshold_callable_out_of_range_raises(bad_value):
    with pytest.raises(TypeError, match="must be a callable or a float"):
        make_correlation_threshold_callable(bad_value)


def test_make_correlation_threshold_callable_non_numeric_raises():
    with pytest.raises(TypeError, match="must be a callable or a float"):
        make_correlation_threshold_callable("not a number")


# --------------------------------------------------------------------------- #
# CorrelationThresholding
# --------------------------------------------------------------------------- #


def test_correlation_thresholding_default_uses_default_threshold():
    ct = CorrelationThresholding()
    assert ct.correlation_threshold is default_correlation_threshold


def test_correlation_thresholding_with_float():
    ct = CorrelationThresholding(0.5)
    assert ct.correlation_threshold(999) == 0.5


def test_correlation_thresholding_with_callable():
    def custom(ne):
        return 0.2

    ct = CorrelationThresholding(custom)
    assert ct.correlation_threshold is custom


def test_correlation_thresholding_call_zeroes_below_threshold():
    ct = CorrelationThresholding(0.5)
    corr = np.array([[1.0, 0.4], [0.6, -0.5]])
    result = ct(corr, ne=100)
    expected = np.array([[1.0, 0.0], [0.6, 0.0]])  # 0.4 and -0.5 zeroed
    np.testing.assert_allclose(result, expected)


# --------------------------------------------------------------------------- #
# CorrelationTempering
# --------------------------------------------------------------------------- #


def test_correlation_tempering_requires_callable():
    with pytest.raises(AssertionError):
        CorrelationTempering(tempering_function="not callable")  # type: ignore[arg-type]


def test_correlation_tempering_call_delegates_to_function():
    calls = []

    def tempering_function(corr, ne):
        calls.append((corr.shape, ne))
        return corr * 0.5

    ct = CorrelationTempering(tempering_function)
    corr = np.eye(3)
    result = ct(corr, ne=42)
    np.testing.assert_allclose(result, corr * 0.5)
    assert calls == [((3, 3), 42)]


# --------------------------------------------------------------------------- #
# CorrelationBasedLocalization
# --------------------------------------------------------------------------- #


def test_correlation_based_localization_init():
    transform = CorrelationThresholding(0.0)  # threshold 0 keeps everything
    loc = CorrelationBasedLocalization(transform)
    assert loc.transform is transform


def test_correlation_based_localization_localize(ensembles):
    X, Y = ensembles
    transform = CorrelationThresholding(0.0)  # keep everything -> no masking
    loc = CorrelationBasedLocalization(transform)
    result = loc.localize(X, Y)
    assert result.shape == (X.shape[0], Y.shape[0])


def test_correlation_based_localization_zeroes_weak_correlations(ensembles):
    X, Y = ensembles
    # Threshold of 1.0 -> nothing can exceed it -> everything zeroed.
    transform = CorrelationThresholding(1.0)
    loc = CorrelationBasedLocalization(transform)
    result = loc.localize(X, Y)
    np.testing.assert_allclose(result, np.zeros_like(result))


def test_correlation_based_localization_multi_dot_no_args(ensembles):
    X, Y = ensembles
    loc = CorrelationBasedLocalization(CorrelationThresholding(0.0))
    result = loc.localize_multi_dot(X, Y)
    ref = loc.localize(X, Y)
    np.testing.assert_allclose(result, ref)


def test_correlation_based_localization_multi_dot_with_args(ensembles, rng):
    X, Y = ensembles
    loc = CorrelationBasedLocalization(CorrelationThresholding(0.0))
    K = rng.random((Y.shape[0], 2))
    result = loc.localize_multi_dot(X, Y, K)
    ref = loc.localize(X, Y) @ K
    np.testing.assert_allclose(result, ref)


# --------------------------------------------------------------------------- #
# _reversed_beta_cumulative
# --------------------------------------------------------------------------- #


def test_reversed_beta_cumulative_negative_beta_raises():
    with pytest.raises(ValueError, match=r"Beta \(-1.0\) should be positive or null"):
        _reversed_beta_cumulative(np.linspace(0, 1, 10), beta=-1.0)


def test_reversed_beta_cumulative_beta_zero_allowed():
    result = _reversed_beta_cumulative(np.linspace(0, 1, 10), beta=0.0)
    assert result[0] == 1.0
    assert result[-1] == 0.0


def test_reversed_beta_cumulative_boundaries():
    result = _reversed_beta_cumulative(np.array([0.0, 0.5, 1.0, 1.5]), beta=3.0)
    assert result[0] == 1.0  # d == 0
    assert result[2] == 0.0  # d == 1 (boundary, masked to nan internally)
    assert result[3] == 0.0  # d > 1


# --------------------------------------------------------------------------- #
# gc_correlation_tempering / gc_correlation_tempering_positive
# --------------------------------------------------------------------------- #


def test_gc_correlation_tempering_small_ensemble_raises():
    with pytest.raises(ValueError, match=r"Cannot use the Gaspari-Cohn tempering"):
        gc_correlation_tempering(np.eye(3), ne=9)
    with pytest.raises(ValueError):
        gc_correlation_tempering(np.eye(3), ne=5)


def test_gc_correlation_tempering_basic():
    corr = np.array([[1.0, 0.9], [0.9, 1.0]])
    result = gc_correlation_tempering(corr, ne=20)
    assert result.shape == corr.shape
    # Perfectly correlated entries (|corr|==1) should map close to weight 1.
    np.testing.assert_allclose(np.diag(result), [1.0, 1.0], atol=1e-8)


def test_gc_correlation_tempering_positive_zeroes_negative_entries():
    corr = np.array([[1.0, -0.9], [-0.9, 1.0]])
    result = gc_correlation_tempering_positive(corr, ne=20)
    assert result[0, 1] == 0.0
    assert result[1, 0] == 0.0
    np.testing.assert_allclose(np.diag(result), [1.0, 1.0], atol=1e-8)


def test_gc_correlation_tempering_positive_small_ensemble_raises():
    with pytest.raises(ValueError):
        gc_correlation_tempering_positive(np.eye(3), ne=9)


# --------------------------------------------------------------------------- #
# distances_to_weights_beta_cumulative (kept from the original test file,
# unchanged, plus the module still needs these two tests co-located so the
# whole file remains self-contained)
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# _part1 / _part2 (Gaspari-Cohn fifth-order polynomial pieces)
# --------------------------------------------------------------------------- #


def test_part1_at_zero_is_one():
    assert _part1(0.0) == 1.0


def test_part1_part2_continuous_at_one():
    # The two polynomial pieces must agree at the z=1 boundary, per the
    # piecewise definition in distances_to_weights_fifth_order's docstring.
    np.testing.assert_allclose(_part1(1.0), _part2(1.0), atol=1e-12)


def test_part2_at_two_is_zero():
    np.testing.assert_allclose(_part2(2.0), 0.0, atol=1e-12)


def test_part1_part2_vectorized():
    d = np.array([0.0, 0.5, 1.0])
    r1 = _part1(d)
    assert r1.shape == d.shape
    d2 = np.array([1.0, 1.5, 2.0])
    r2 = _part2(d2)
    assert r2.shape == d2.shape


# --------------------------------------------------------------------------- #
# distances_to_weights_fifth_order (kept from the original test file, plus
# extra cases to reach the negative-distance and boundary branches)
# --------------------------------------------------------------------------- #


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


def test_distances_to_weights_fifth_order_negative_distance_masked_to_zero():
    # distances2 < 0 is an explicitly-handled branch (distances2[distances2
    # < 0] = np.nan), even though physical distances are never negative in
    # normal use.
    result = distances_to_weights_fifth_order(np.array([-5.0, 0.0, 0.5]))
    assert result[0] == 0.0


def test_distances_to_weights_fifth_order_far_beyond_support_is_zero():
    # distances2 >= 2.0 -> masked to nan -> weight 0.
    result = distances_to_weights_fifth_order(np.array([10.0]), scaling_factor=1.0)
    assert result[0] == 0.0


def test_distances_to_weights_fifth_order_boundary_at_one():
    # z == 1.0 is the boundary between the two polynomial pieces; both
    # should agree here (see test_part1_part2_continuous_at_one), so the
    # overall function should be continuous through it too.
    result = distances_to_weights_fifth_order(np.array([0.99, 1.0, 1.01]))
    assert np.all(np.diff(result) <= 1e-6)  # monotonically non-increasing
