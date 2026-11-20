"""
三合一测试：正确性 + 性能 + NumPy 对比
"""
import time
import numpy as np
import rvv

def test_vector_correct():
    """1. 向量运算正确性"""
    a = np.array([1, 2, 3, 4], dtype=np.float32)
    b = np.array([5, 6, 7, 8], dtype=np.float32)
    assert np.allclose(rvv.add(a, b), a + b)
    assert np.allclose(rvv.sub(a, b), a - b)
    assert np.allclose(rvv.scale(a, 3.0), a * 3)
    assert abs(rvv.dot(a, b) - np.dot(a, b)) < 1e-6
    assert abs(rvv.norm_l2(a) - np.linalg.norm(a)) < 1e-6
    print("✓ vector correctness passed")

def test_matrix_correct():
    """2. 矩阵运算正确性"""
    A = np.arange(12, dtype=np.float32).reshape(3, 4)
    B = np.arange(12, 24, dtype=np.float32).reshape(4, 3)
    C = rvv.matmul(A, B)
    C_np = A @ B
    assert np.allclose(C, C_np)
    assert np.allclose(rvv.transpose(A), A.T)
    print("✓ matrix correctness passed")

def test_performance():
    """3. 性能对比（大向量）"""
    n = 1_000_000
    a = np.random.rand(n).astype(np.float32)
    b = np.random.rand(n).astype(np.float32)

    t0 = time.time()
    rvv.add(a, b)
    t_rvv = time.time() - t0

    t0 = time.time()
    a + b
    t_np = time.time() - t0

    speedup = t_np / (t_rvv + 1e-6)
    print(f"add  1M float32:  NumPy {t_np*1000:.2f} ms  vs  rvv {t_rvv*1000:.2f} ms  "
          f"→ speedup {speedup:.2f}×")

if __name__ == "__main__":
    test_vector_correct()
    test_matrix_correct()
    test_performance()
    print("All tests passed!")