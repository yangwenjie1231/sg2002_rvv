"""Device regression suite for rvv on SG2002/C906.

Runs on the MaixCAM itself. Compares every operator against a pure-Python
reference so RVV-only bugs cannot hide behind a passing scalar build.
Exit code 0 == all checks passed.
"""
import math
import sys
import time

failed = []
passed = 0


def check(name, cond, detail=""):
    global passed
    if cond:
        passed += 1
        print("PASS %s" % name)
    else:
        failed.append(name)
        print("FAIL %s %s" % (name, detail))


def close(a, b, rel=1e-4, abs_=1e-4):
    return abs(a - b) <= max(abs_, abs(b) * rel)


def run():
    import rvv

    # ---- vector elementwise ----
    for n in (1, 3, 4, 7, 8, 9, 15, 16, 17, 33, 100, 1000, 5000):
        a = [float(i + 1) for i in range(n)]
        b = [float((i % 7) + 1) for i in range(n)]
        got = rvv.add(a, b)
        check("add n=%d" % n, all(close(float(g), x + y) for g, x, y in zip(got, a, b)))
        got = rvv.sub(a, b)
        check("sub n=%d" % n, all(close(float(g), x - y) for g, x, y in zip(got, a, b)))
        got = rvv.scale(a, 2.5)
        check("scale n=%d" % n, all(close(float(g), x * 2.5) for g, x in zip(got, a)))
        got = rvv.mul_f32(a, b)
        check("mul_f32 n=%d" % n, all(close(float(g), x * y) for g, x, y in zip(got, a, b)))
        got = rvv.add_scalar_f32(a, 3.0)
        check("add_scalar n=%d" % n, all(close(float(g), x + 3.0) for g, x in zip(got, a)))
        got = rvv.abs_f32([-x for x in a])
        check("abs_f32 n=%d" % n, all(close(float(g), x) for g, x in zip(got, a)))
        got = rvv.clip_f32(a, 2.0, 5.0)
        check("clip_f32 n=%d" % n,
              all(close(float(g), min(max(x, 2.0), 5.0)) for g, x in zip(got, a)))

        # ---- reductions ----
        exp_dot = sum(x * y for x, y in zip(a, b))
        check("dot n=%d" % n, close(rvv.dot(a, b), exp_dot, rel=2e-4),
              "got %r expect %r" % (rvv.dot(a, b), exp_dot))
        exp_sum = sum(a)
        check("sum_f32 n=%d" % n, close(rvv.sum_f32(a), exp_sum, rel=2e-4),
              "got %r expect %r" % (rvv.sum_f32(a), exp_sum))
        exp_norm = math.sqrt(sum(x * x for x in a))
        check("norm_l2 n=%d" % n, close(rvv.norm_l2(a), exp_norm, rel=2e-4),
              "got %r expect %r" % (rvv.norm_l2(a), exp_norm))
        exp_max = max(a)
        check("max_f32 n=%d" % n, close(rvv.max_f32(a), exp_max, rel=2e-4),
              "got %r expect %r" % (rvv.max_f32(a), exp_max))
        exp_min = min(a)
        check("min_f32 n=%d" % n, close(rvv.min_f32(a), exp_min, rel=2e-4),
              "got %r expect %r" % (rvv.min_f32(a), exp_min))
        mean = exp_sum / n
        var = sum((x - mean) ** 2 for x in a) / n
        got_mean, got_std = rvv.mean_std_f32(a)
        check("mean_std mean n=%d" % n, close(got_mean, mean, rel=2e-4),
              "got %r expect %r" % (got_mean, mean))
        check("mean_std std n=%d" % n, close(got_std, math.sqrt(var), rel=1e-3),
              "got %r expect %r" % (got_std, math.sqrt(var)))

    # ---- NaN semantics ----
    check("max NaN", math.isnan(rvv.max_f32([1.0, float("nan"), 3.0])))
    check("min NaN", math.isnan(rvv.min_f32([1.0, float("nan"), 3.0])))
    check("max empty", math.isnan(rvv.max_f32([])))
    check("min empty", math.isnan(rvv.min_f32([])))
    check("sum empty", rvv.sum_f32([]) == 0.0)
    check("dot empty", rvv.dot([], []) == 0.0)

    # ---- matrices ----
    for (m, k, nn) in ((1, 1, 1), (2, 3, 4), (4, 4, 4), (3, 5, 2), (8, 8, 8)):
        A = [[float((i * k + j) % 5) for j in range(k)] for i in range(m)]
        B = [[float((i * nn + j) % 3) for j in range(nn)] for i in range(k)]
        C = [[sum(A[i][t] * B[t][j] for t in range(k)) for j in range(nn)] for i in range(m)]
        got = rvv.matmul(A, B)
        ok = all(close(float(got[i][j]), C[i][j], rel=2e-4) for i in range(m) for j in range(nn))
        check("matmul %dx%dx%d" % (m, k, nn), ok)

    print("\n%d passed, %d failed" % (passed, len(failed)))
    if failed:
        print("FAILED:", failed)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run())
