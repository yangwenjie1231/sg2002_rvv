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

    # ---- gradient_stats_u8 vs pure Python ----
    def reference(buf, width, height, stride, roi_fraction=0.7,
                  step_x=0, step_y=0, offset=2, channels=3):
        if width < 4 or height < 4 or offset < 1:
            return dict(grad_sum=0.0, lum_sum=0.0, lum_sq_sum=0.0,
                        count=0, saturated_count=0)
        rf = min(max(float(roi_fraction), 0.1), 1.0)
        rw = max(4, int(width * rf))
        rh = max(4, int(height * rf))
        x0 = (width - rw) // 2
        y0 = (height - rh) // 2
        if step_x == 0:
            step_x = max(2, rw // 100)
        if step_y == 0:
            step_y = max(2, rh // 100)

        def gray(x, y):
            o = y * stride + x * channels
            if channels == 1:
                return float(buf[o])
            return (float(buf[o]) + float(buf[o + 1]) + float(buf[o + 2])) / 3.0

        gs = ls = lq = 0.0
        sat = 0
        cnt = 0
        for y in range(y0, y0 + rh - offset, step_y):
            for x in range(x0, x0 + rw - offset, step_x):
                c = gray(x, y)
                cx = gray(x + offset, y)
                cy = gray(x, y + offset)
                dx = cx - c
                dy = cy - c
                gs += dx * dx + dy * dy
                ls += c
                lq += c * c
                if c >= 250.0:
                    sat += 1
                cnt += 1
        return dict(grad_sum=gs, lum_sum=ls, lum_sq_sum=lq,
                    count=cnt, saturated_count=sat)

    cases = []
    for (w, h) in ((64, 48), (160, 120), (320, 240), (640, 480)):
        rgb = bytearray(w * h * 3)
        for y in range(h):
            for x in range(w):
                o = (y * w + x) * 3
                rgb[o] = (x * 7 + y * 3) % 256
                rgb[o + 1] = (x * 2 + y * 11) % 256
                rgb[o + 2] = (x * 13 + y * 5) % 256
        # saturate a block so saturated_count is exercised
        for y in range(h // 4, h // 3):
            for x in range(w // 4, w // 3):
                o = (y * w + x) * 3
                rgb[o:o + 3] = b"\xff\xff\xff"
        graybuf = bytearray((x * 11 + y * 7 + (x * y) % 23) % 256
                            for y in range(h) for x in range(w))
        cases = [
            ("rgb", bytes(rgb), w, h, w * 3, dict(channels=3)),
            ("rgb roi1", bytes(rgb), w, h, w * 3, dict(channels=3, roi_fraction=1.0)),
            ("rgb roi.5", bytes(rgb), w, h, w * 3, dict(channels=3, roi_fraction=0.5)),
            ("rgb step", bytes(rgb), w, h, w * 3, dict(channels=3, step_x=3, step_y=2)),
            ("rgb off3", bytes(rgb), w, h, w * 3, dict(channels=3, offset=3)),
            ("gray", bytes(graybuf), w, h, w, dict(channels=1)),
            ("gray step", bytes(graybuf), w, h, w, dict(channels=1, step_x=5, step_y=4)),
        ]
        for (label, buf, ww, hh, st, cfg) in cases:
            got = rvv.gradient_stats_u8(buf, ww, hh, st, **cfg)
            exp = reference(buf, ww, hh, st, **cfg)
            ok = (got["count"] == exp["count"]
                  and got["saturated_count"] == exp["saturated_count"]
                  and close(got["grad_sum"], exp["grad_sum"], rel=1e-3, abs_=1e-2)
                  and close(got["lum_sum"], exp["lum_sum"], rel=1e-3, abs_=1e-2)
                  and close(got["lum_sq_sum"], exp["lum_sq_sum"], rel=1e-3, abs_=1e-2))
            check("gradient %dx%d %s" % (w, h, label), ok,
                  "\n   got %r\n   exp %r" % (got, exp))

    # ---- edge cases ----
    check("gradient empty", rvv.gradient_stats_u8(b"", 8, 8, 8, channels=1)["count"] == 0)
    check("gradient tiny", rvv.gradient_stats_u8(bytearray(9), 3, 3, 3, channels=1)["count"] == 0)

    # ---- benchmark: gradient_stats_u8 vs pure python ----
    w, h = 640, 480
    buf = bytes(bytearray((x * 3 + y * 5) % 256 for y in range(h) for x in range(w * 3)))
    t0 = time.time()
    for _ in range(20):
        rvv.gradient_stats_u8(buf, w, h, w * 3, channels=3)
    rvv_ms = (time.time() - t0) / 20 * 1000
    print("BENCH gradient_stats_u8 640x480 rgb: %.2f ms/call" % rvv_ms)

    print("\n%d passed, %d failed" % (passed, len(failed)))
    if failed:
        print("FAILED:", failed)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run())