import pytest

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None

import rvv


@pytest.mark.skipif(np is None, reason="NumPy not available on this environment")
def test_add_sub_scale_dot_norm():
    n = 257  # odd to exercise tail handling
    a = np.random.RandomState(0).randn(n).astype(np.float32)
    b = np.random.RandomState(1).randn(n).astype(np.float32)

    np.testing.assert_allclose(rvv.add(a, b), a + b, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(rvv.sub(a, b), a - b, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(rvv.scale(a, 0.25), a * np.float32(0.25), rtol=1e-6, atol=1e-6)

    got_dot = rvv.dot(a, b)
    ref_dot = float(np.dot(a, b))
    assert abs(got_dot - ref_dot) / (abs(ref_dot) + 1e-6) < 1e-5

    got_norm = rvv.norm_l2(a)
    ref_norm = float(np.linalg.norm(a))
    assert abs(got_norm - ref_norm) / (abs(ref_norm) + 1e-6) < 1e-5


@pytest.mark.skipif(np is None, reason="NumPy not available on this environment")
def test_matmul():
    rs = np.random.RandomState(0)
    A = rs.randn(17, 23).astype(np.float32)
    B = rs.randn(23, 9).astype(np.float32)
    C = rvv.matmul(A, B)
    ref = A @ B
    np.testing.assert_allclose(C, ref, rtol=1e-4, atol=1e-4)
