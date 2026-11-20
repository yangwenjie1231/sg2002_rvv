#!/usr/bin/env python3
"""Tests for the float32 extension operators."""

import math
import numpy as np
import rvv


def _expect_value_error(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_abs_f32():
    a = np.array([-3.5, -0.0, 0.0, 2.25, -7.0], dtype=np.float32)
    out = rvv.abs_f32(a)
    assert out.dtype == np.float32
    assert np.allclose(out, np.abs(a))
    assert not np.signbit(out[1])

    assert rvv.abs_f32(np.array([], dtype=np.float32)).size == 0
    large = np.linspace(-1e4, 1e4, 1_000_003, dtype=np.float32)
    large_out = rvv.abs_f32(large)
    assert np.array_equal(large_out, np.abs(large))

    # Sliced views must be rejected, not read as if they were contiguous.
    try:
        rvv.abs_f32(np.arange(10, dtype=np.float32)[::2])
    except ValueError:
        pass
    else:
        raise AssertionError("non-contiguous float32 input should be rejected")


def test_sum_f32():
    rng = np.random.default_rng(1234)
    a = rng.standard_normal(4099).astype(np.float32)
    expected = float(np.sum(a, dtype=np.float64))
    actual = rvv.sum_f32(a)
    assert math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-5)
    assert rvv.sum_f32(np.array([], dtype=np.float32)) == 0.0

    # Large deterministic input, including negative values (cancellation).
    large = np.linspace(-1.0, 1.0, 1_000_003, dtype=np.float32)
    expected_large = float(np.sum(large, dtype=np.float64))
    actual_large = rvv.sum_f32(large)
    assert math.isclose(actual_large, expected_large,
                        rel_tol=1e-6, abs_tol=1e-5)


def test_mean_std_f32():
    rng = np.random.default_rng(42)
    a = rng.normal(10.0, 3.0, size=10007).astype(np.float32)
    mean, std = rvv.mean_std_f32(a)
    assert math.isclose(mean, float(np.mean(a, dtype=np.float64)),
                        rel_tol=1e-5, abs_tol=1e-5)
    assert math.isclose(std, float(np.std(a, dtype=np.float64)),
                        rel_tol=1e-5, abs_tol=1e-5)

    large = np.linspace(-100.0, 100.0, 200_003, dtype=np.float32)
    mean, std = rvv.mean_std_f32(large)
    assert math.isclose(mean, float(np.mean(large, dtype=np.float64)),
                        rel_tol=1e-6, abs_tol=1e-5)
    assert math.isclose(std, float(np.std(large, dtype=np.float64)),
                        rel_tol=1e-6, abs_tol=1e-5)

    # Single element and all-equal input have zero population std.
    assert rvv.mean_std_f32(np.array([4.5], dtype=np.float32)) == (4.5, 0.0)
    constant = rvv.mean_std_f32(np.full(1000, -3.25, dtype=np.float32))
    assert math.isclose(constant[0], -3.25, rel_tol=0.0, abs_tol=0.0)
    assert constant[1] == 0.0
    # Empty input follows the documented NaN contract.
    empty = rvv.mean_std_f32(np.array([], dtype=np.float32))
    assert math.isnan(empty[0]) and math.isnan(empty[1])


def test_mul_f32():
    a = np.array([-2.0, -0.5, 0.0, 1.5, 4.0], dtype=np.float32)
    b = np.array([3.0, 4.0, 9.0, -2.0, 0.25], dtype=np.float32)
    out = rvv.mul_f32(a, b)
    assert out.dtype == np.float32
    assert np.array_equal(out, a * b)

    empty = rvv.mul_f32(
        np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    )
    assert empty.size == 0

    _expect_value_error(lambda: rvv.mul_f32(
        np.ones(3, dtype=np.float32), np.ones(4, dtype=np.float32)
    ))
    _expect_value_error(lambda: rvv.mul_f32(
        np.arange(10, dtype=np.float32)[::2],
        np.arange(5, dtype=np.float32),
    ))

    rng = np.random.default_rng(2026)
    large_a = rng.standard_normal(1_000_003).astype(np.float32)
    large_b = rng.standard_normal(1_000_003).astype(np.float32)
    np.testing.assert_allclose(
        rvv.mul_f32(large_a, large_b), large_a * large_b,
        rtol=1e-6, atol=1e-6,
    )


def test_add_scalar_f32():
    a = np.array([-2.0, 0.0, 3.5], dtype=np.float32)
    out = rvv.add_scalar_f32(a, 1.25)
    assert out.dtype == np.float32
    assert np.array_equal(out, a + np.float32(1.25))
    assert rvv.add_scalar_f32(
        np.array([], dtype=np.float32), 7.0
    ).size == 0

    # NaN input propagates naturally through addition.
    nan_out = rvv.add_scalar_f32(
        np.array([np.nan], dtype=np.float32), 1.0
    )
    assert math.isnan(float(nan_out[0]))

    rng = np.random.default_rng(7)
    large = rng.uniform(-1e4, 1e4, 500_003).astype(np.float32)
    np.testing.assert_allclose(
        rvv.add_scalar_f32(large, -3.5), large - np.float32(3.5),
        rtol=1e-6, atol=1e-5,
    )


def test_clip_f32():
    a = np.array([-np.inf, -3.0, 0.0, 2.0, np.inf, np.nan],
                 dtype=np.float32)
    out = rvv.clip_f32(a, -1.0, 1.0)
    expected = np.array([-1.0, -1.0, 0.0, 1.0, 1.0, np.nan],
                        dtype=np.float32)
    assert np.array_equal(out[:-1], expected[:-1])
    assert math.isnan(float(out[-1]))

    assert rvv.clip_f32(
        np.array([], dtype=np.float32), 0.0, 1.0
    ).size == 0
    np.testing.assert_array_equal(
        rvv.clip_f32(a, -np.inf, np.inf), a
    )
    _expect_value_error(lambda: rvv.clip_f32(a, 2.0, 1.0))
    _expect_value_error(lambda: rvv.clip_f32(a, np.nan, 1.0))
    _expect_value_error(lambda: rvv.clip_f32(
        np.arange(10, dtype=np.float32)[::2], 0.0, 1.0
    ))


def test_max_f32():
    a = np.array([-3.5, 7.25, 0.0, -0.0], dtype=np.float32)
    assert rvv.max_f32(a) == np.max(a)
    assert rvv.max_f32(np.array([4.5], dtype=np.float32)) == 4.5
    assert math.isnan(rvv.max_f32(np.array([], dtype=np.float32)))
    assert math.isnan(rvv.max_f32(
        np.array([1.0, np.nan, 3.0], dtype=np.float32)
    ))

    rng = np.random.default_rng(99)
    large = rng.standard_normal(1_000_003).astype(np.float32)
    assert rvv.max_f32(large) == np.max(large)


def test_min_f32():
    a = np.array([-3.5, 7.25, 0.0, -0.0], dtype=np.float32)
    assert rvv.min_f32(a) == np.min(a)
    assert rvv.min_f32(np.array([4.5], dtype=np.float32)) == 4.5
    assert math.isnan(rvv.min_f32(np.array([], dtype=np.float32)))
    assert math.isnan(rvv.min_f32(
        np.array([1.0, np.nan, 3.0], dtype=np.float32)
    ))
    _expect_value_error(lambda: rvv.max_f32(
        np.arange(10, dtype=np.float32)[::2]
    ))
    _expect_value_error(lambda: rvv.min_f32(
        np.arange(10, dtype=np.float32)[::2]
    ))

    rng = np.random.default_rng(100)
    large = rng.standard_normal(1_000_003).astype(np.float32)
    assert rvv.min_f32(large) == np.min(large)


def main():
    tests = [
        test_abs_f32,
        test_sum_f32,
        test_mean_std_f32,
        test_mul_f32,
        test_add_scalar_f32,
        test_clip_f32,
        test_max_f32,
        test_min_f32,
    ]
    for fn in tests:
        fn()
        print("PASS {}".format(fn.__name__))
    print("All {} extension-op tests passed.".format(len(tests)))


if __name__ == "__main__":
    main()
