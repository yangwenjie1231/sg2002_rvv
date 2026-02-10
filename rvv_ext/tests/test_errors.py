import pytest

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None

import rvv


@pytest.mark.skipif(np is None, reason="NumPy not available on this environment")
def test_dtype_is_forcecasted():
    a = np.arange(8, dtype=np.float64)
    b = np.arange(8, dtype=np.float64)
    out = rvv.add(a, b)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, (a + b).astype(np.float32), rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(np is None, reason="NumPy not available on this environment")
def test_shape_error_dot():
    a = np.arange(8, dtype=np.float32)
    b = np.arange(9, dtype=np.float32)
    with pytest.raises(ValueError, match=r"same length"):
        rvv.dot(a, b)


@pytest.mark.skipif(np is None, reason="NumPy not available on this environment")
def test_non_contiguous_is_copied():
    a = np.arange(32, dtype=np.float32)[::2]
    b = np.ones_like(a)
    out = rvv.add(a, b)
    np.testing.assert_allclose(out, a + b, rtol=1e-6, atol=1e-6)
