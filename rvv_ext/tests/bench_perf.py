"""Tiny benchmark.

On SG2002 you should see rvv (RVV path) noticeably faster than NumPy for simple ops
when arrays are large enough and CPU isn't throttled.

This script prints times; it's not a unit test.
"""

from __future__ import annotations

import time
import numpy as np

import rvv


def bench(name, fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    dt = (t1 - t0) / iters
    print(f"{name:20s}: {dt*1000:.3f} ms")


def main():
    n = 1_000_000
    a = np.random.RandomState(0).randn(n).astype(np.float32)
    b = np.random.RandomState(1).randn(n).astype(np.float32)

    bench("rvv.add", lambda: rvv.add(a, b), iters=20)
    bench("numpy add", lambda: a + b, iters=20)

    bench("rvv.dot", lambda: rvv.dot(a, b), iters=10)
    bench("numpy dot", lambda: float(np.dot(a, b)), iters=10)

    A = np.random.RandomState(2).randn(256, 256).astype(np.float32)
    B = np.random.RandomState(3).randn(256, 256).astype(np.float32)
    bench("rvv.matmul", lambda: rvv.matmul(A, B), iters=3, warmup=1)
    bench("numpy matmul", lambda: A @ B, iters=3, warmup=1)


if __name__ == "__main__":
    main()
