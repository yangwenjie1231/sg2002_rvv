"""
int8 正确性 + 性能测试
"""
import time
import numpy as np
import rvv

def test_int8_vector():
    a = np.array([1, 2, 3, 4], dtype=np.int8)
    b = np.array([5, 6, 7, 8], dtype=np.int8)
    assert np.allclose(rvv.add_i8(a, b), a + b)
    assert rvv.dot_i8(a, b) == int(np.dot(a.astype(int), b.astype(int)))
    print("✓ int8 vector functions passed")

def test_int8_matrix():
    A = np.arange(12, dtype=np.int8).reshape(3, 4)
    B = np.arange(12, 24, dtype=np.int8).reshape(3, 4)
    C = rvv.add2d_i8(A, B)
    assert np.allclose(C, A + B)
    print("✓ int8 matrix functions passed")

def test_int8_performance():
    n = 1_000_000
    a = np.random.randint(-10, 10, size=n, dtype=np.int8)
    b = np.random.randint(-10, 10, size=n, dtype=np.int8)

    t0 = time.time()
    rvv.dot_i8(a, b)
    t_rvv = time.time() - t0

    t0 = time.time()
    np.dot(a.astype(int), b.astype(int))
    t_np = time.time() - t0

    speedup = t_np / (t_rvv + 1e-6)
    print(f"dot_i8 1M int8:  NumPy {t_np*1000:.2f} ms  vs  rvv {t_rvv*1000:.2f} ms  "
          f"→ speedup {speedup:.2f}×")

if __name__ == "__main__":
    test_int8_vector()
    test_int8_matrix()
    test_int8_performance()
    print("All int8 tests passed!")