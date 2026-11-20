#!/usr/bin/env python3
"""gradient_stats_u8 correctness tests.

This file intentionally does not import NumPy: it uses bytearray/memoryview
inputs and a pure-Python reference implementation. Run directly:

    python tests/test_gradient_stats.py

or with pytest:

    pytest -q tests/test_gradient_stats.py
"""

import math
import rvv

ZERO_STATS = {
    "grad_sum": 0.0,
    "lum_sum": 0.0,
    "lum_sq_sum": 0.0,
    "count": 0,
    "saturated_count": 0,
}


def _gray(buf, stride, x, y, channels):
    off = y * stride + x * channels
    if channels == 1:
        return float(buf[off])
    return (float(buf[off]) + float(buf[off + 1]) + float(buf[off + 2])) / 3.0


def reference_gradient_stats(buf, width, height, stride,
                             roi_fraction=0.7, step_x=0, step_y=0,
                             offset=2, channels=3):
    """Line-by-line port of firmware/utils/image_utils.py diagnostics loop."""
    if width < 4 or height < 4 or offset < 1:
        return dict(ZERO_STATS)

    roi_fraction = min(max(float(roi_fraction), 0.1), 1.0)
    roi_w = max(4, int(width * roi_fraction))
    roi_h = max(4, int(height * roi_fraction))
    x0 = (width - roi_w) // 2
    y0 = (height - roi_h) // 2
    if step_x == 0:
        step_x = max(2, roi_w // 100)
    if step_y == 0:
        step_y = max(2, roi_h // 100)

    grad_sum = 0.0
    lum_sum = 0.0
    lum_sq_sum = 0.0
    saturated_count = 0
    count = 0

    for y in range(y0, y0 + roi_h - offset, step_y):
        for x in range(x0, x0 + roi_w - offset, step_x):
            c = _gray(buf, stride, x, y, channels)
            cx = _gray(buf, stride, x + offset, y, channels)
            cy = _gray(buf, stride, x, y + offset, channels)
            dx = cx - c
            dy = cy - c
            grad_sum += dx * dx + dy * dy
            lum_sum += c
            lum_sq_sum += c * c
            if c >= 250.0:
                saturated_count += 1
            count += 1

    return {
        "grad_sum": grad_sum,
        "lum_sum": lum_sum,
        "lum_sq_sum": lum_sq_sum,
        "count": count,
        "saturated_count": saturated_count,
    }


def _assert_close(actual, expected, rel=2e-6, abs_=1e-6):
    if actual == expected:
        return
    if not math.isclose(actual, expected, rel_tol=rel, abs_tol=abs_):
        raise AssertionError("{} != {} (rel={}, abs={})".format(
            actual, expected, rel, abs_))


def check_stats(actual, expected, rel=2e-6, abs_=1e-6):
    assert isinstance(actual, dict)
    assert set(actual) == set(ZERO_STATS)
    assert isinstance(actual["count"], int)
    assert isinstance(actual["saturated_count"], int)
    assert isinstance(actual["grad_sum"], float)
    assert isinstance(actual["lum_sum"], float)
    assert isinstance(actual["lum_sq_sum"], float)
    assert actual["count"] == expected["count"]
    assert actual["saturated_count"] == expected["saturated_count"]
    for key in ("grad_sum", "lum_sum", "lum_sq_sum"):
        _assert_close(actual[key], expected[key], rel=rel, abs_=abs_)


def make_gray_pattern(width, height):
    buf = bytearray(width * height)
    for y in range(height):
        for x in range(width):
            value = (x * 11 + y * 7 + (x * y) % 23) % 256
            if 20 <= x < 27 and 12 <= y < 18:
                value = 255
            buf[y * width + x] = value
    return buf


def make_padded_gray(width, height, stride, pad=0xA5):
    raw = bytearray(stride * height)
    for y in range(height):
        for x in range(width):
            raw[y * stride + x] = (x * 5 + y * 13 + 17) % 256
        for x in range(width, stride):
            raw[y * stride + x] = pad
    return raw


def test_gray_reference():
    width, height = 61, 47
    buf = make_gray_pattern(width, height)
    expected = reference_gradient_stats(
        buf, width, height, width, roi_fraction=0.7, channels=1
    )
    actual = rvv.gradient_stats_u8(
        buf, width, height, width, roi_fraction=0.7, channels=1
    )
    check_stats(actual, expected)


def test_rgb_and_gray_equivalence():
    width, height = 53, 39
    gray = make_gray_pattern(width, height)
    rgb = bytearray(width * height * 3)
    for i, value in enumerate(gray):
        rgb[i * 3:i * 3 + 3] = bytes((value, value, value))

    gray_stats = rvv.gradient_stats_u8(
        gray, width, height, width, channels=1
    )
    rgb_stats = rvv.gradient_stats_u8(
        memoryview(rgb), width, height, width * 3, channels=3
    )
    check_stats(rgb_stats, gray_stats)
    assert rgb_stats["count"] == gray_stats["count"]


def test_color_rgb_reference():
    width, height = 64, 48
    buf = bytearray(width * height * 3)
    for y in range(height):
        for x in range(width):
            off = (y * width + x) * 3
            buf[off] = (x * 7 + y * 3) % 256
            buf[off + 1] = (x * 2 + y * 11) % 256
            buf[off + 2] = (x * 13 + y * 5) % 256
    # Saturate a region to exercise saturated_count (not any-channel logic).
    for y in range(10, 20):
        for x in range(15, 25):
            off = (y * width + x) * 3
            buf[off:off + 3] = b"\xff\xff\xff"

    expected = reference_gradient_stats(
        buf, width, height, width * 3,
        roi_fraction=0.9, step_x=3, step_y=2, offset=3, channels=3
    )
    actual = rvv.gradient_stats_u8(
        bytes(buf), width, height, width * 3,
        roi_fraction=0.9, step_x=3, step_y=2, offset=3, channels=3
    )
    check_stats(actual, expected)
    assert expected["saturated_count"] > 0


def test_shaped_memoryviews():
    width, height = 29, 21
    gray = make_gray_pattern(width, height)
    view_2d = memoryview(bytes(gray)).cast("B", shape=(height, width))
    actual = rvv.gradient_stats_u8(
        view_2d, width, height, width, roi_fraction=0.8,
        step_x=2, step_y=2, offset=2, channels=1
    )
    expected = reference_gradient_stats(
        gray, width, height, width, roi_fraction=0.8,
        step_x=2, step_y=2, offset=2, channels=1
    )
    check_stats(actual, expected)

    rgb = bytearray(width * height * 3)
    for y in range(height):
        for x in range(width):
            value = gray[y * width + x]
            off = (y * width + x) * 3
            rgb[off:off + 3] = bytes((value, value, value))
    view_3d = memoryview(bytes(rgb)).cast(
        "B", shape=(height, width, 3)
    )
    actual_rgb = rvv.gradient_stats_u8(
        view_3d, width, height, width * 3, roi_fraction=0.8,
        step_x=2, step_y=2, offset=2, channels=3
    )
    check_stats(actual_rgb, expected)


def test_explicit_steps_and_offset():
    width, height = 80, 60
    buf = make_padded_gray(width, height, width + 7)
    kwargs = dict(roi_fraction=0.55, step_x=5, step_y=4,
                  offset=2, channels=1)
    expected = reference_gradient_stats(
        buf, width, height, width + 7, **kwargs
    )
    actual = rvv.gradient_stats_u8(
        buf, width, height, width + 7, **kwargs
    )
    check_stats(actual, expected)


def test_roi_fraction_clamped_like_firmware():
    width, height = 32, 24
    buf = make_gray_pattern(width, height)
    for fraction in (0.0, 0.01, -3.0, 1.0):
        expected = reference_gradient_stats(
            buf, width, height, width, roi_fraction=fraction, channels=1
        )
        actual = rvv.gradient_stats_u8(
            buf, width, height, width, roi_fraction=fraction, channels=1
        )
        check_stats(actual, expected)


def test_empty_and_small_images():
    for width, height, stride in ((0, 0, 0), (3, 8, 3), (8, 3, 8)):
        buf = bytearray(width * height)
        expected = dict(ZERO_STATS)
        actual = rvv.gradient_stats_u8(
            buf, width, height, stride, channels=1
        )
        check_stats(actual, expected)

    # A zero-byte buffer is treated as "no samples" even if dimensions were
    # supplied by a caller that has not allocated the frame yet.
    check_stats(
        rvv.gradient_stats_u8(bytearray(), 8, 8, 8, channels=1),
        dict(ZERO_STATS),
    )


def test_offset_exceeds_roi():
    width, height = 16, 16
    buf = bytearray([100]) * (width * height)
    actual = rvv.gradient_stats_u8(
        buf, width, height, width, offset=width, channels=1
    )
    check_stats(actual, dict(ZERO_STATS))


def test_all_saturated():
    width, height = 10, 10
    buf = bytearray([255]) * (width * height)
    stats = rvv.gradient_stats_u8(
        buf, width, height, width, roi_fraction=1.0,
        step_x=2, step_y=2, offset=2, channels=1
    )
    assert stats["grad_sum"] == 0.0
    assert stats["count"] > 0
    assert stats["saturated_count"] == stats["count"]


def test_invalid_inputs():
    width, height = 16, 16
    good = bytearray(width * height)
    cases = [
        dict(buf=good, width=width, height=height, stride=width,
             channels=2),
        dict(buf=good, width=width, height=height, stride=width,
             offset=0, channels=1),
        dict(buf=good, width=width, height=height, stride=width,
             step_x=-1, channels=1),
        dict(buf=good, width=width, height=height, stride=width - 1,
             channels=1),
        dict(buf=bytearray(width), width=width, height=height,
             stride=width, channels=1),
    ]
    for kwargs in cases:
        try:
            rvv.gradient_stats_u8(**kwargs)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError("expected ValueError/TypeError for {}".format(
                kwargs))


def test_non_contiguous_view_rejected():
    width, height, stride = 16, 8, 32
    buf = make_padded_gray(width, height, stride)
    # A strided memoryview has non-unit strides; the API intentionally
    # requires C-contiguous buffers and an explicit row pitch.
    view = memoryview(buf)[::2]
    try:
        rvv.gradient_stats_u8(
            view, width, height, width, channels=1
        )
    except ValueError:
        pass
    else:
        raise AssertionError("non-contiguous buffer should be rejected")


def main():
    tests = [
        test_gray_reference,
        test_rgb_and_gray_equivalence,
        test_color_rgb_reference,
        test_shaped_memoryviews,
        test_explicit_steps_and_offset,
        test_roi_fraction_clamped_like_firmware,
        test_empty_and_small_images,
        test_offset_exceeds_roi,
        test_all_saturated,
        test_invalid_inputs,
        test_non_contiguous_view_rejected,
    ]
    for fn in tests:
        fn()
        print("PASS {}".format(fn.__name__))
    print("All {} gradient_stats_u8 tests passed.".format(len(tests)))


if __name__ == "__main__":
    main()
