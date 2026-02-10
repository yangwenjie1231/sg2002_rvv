"""Type hints for the `rvv` extension module.

This stub file enables IDE auto-completion and type checking.

The runtime implementation is provided by the native extension `rvv.so`.

All array-like inputs are expected to be float32 and C-contiguous.

Note:
- If NumPy is installed, `numpy.ndarray` is the primary way to pass arrays.
- In minimal environments without NumPy, any buffer-protocol object compatible
  with float32 + C-contiguous layout may work.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, Tuple, Union, overload


class _SupportsBuffer(Protocol):
    """A minimal Protocol for buffer-protocol objects.

    Python doesn't expose the buffer protocol as a standard typing protocol.
    This is a lightweight helper so IDEs can accept numpy arrays and other
    buffer objects.
    """

    # No members required; structural typing only.
    pass


# Prefer importing numpy for precise ndarray typing when available.
try:
    import numpy as _np

    NDArrayF32 = _np.ndarray
except Exception:  # pragma: no cover
    NDArrayF32 = Any  # type: ignore


ArrayLike1D = Union[NDArrayF32, _SupportsBuffer]
ArrayLike2D = Union[NDArrayF32, _SupportsBuffer]


def add(a: ArrayLike1D, b: ArrayLike1D) -> NDArrayF32:
    """Elementwise addition of two 1-D float32 vectors.

    Parameters
    ----------
    a, b:
        1-D array-like (buffer-protocol) objects. Must have the same length.
        Expected dtype: float32. Expected layout: C-contiguous.

    Returns
    -------
    numpy.ndarray
        A new float32 array containing ``a + b``.

    Raises
    ------
    ValueError
        If ``a`` and ``b`` have different lengths or are not 1-D.
    TypeError
        If inputs are not float32-compatible buffers.

    Examples
    --------
    >>> import numpy as np, rvv
    >>> a = np.arange(8, dtype=np.float32)
    >>> rvv.add(a, a)
    array([ 0.,  2.,  4.,  6.,  8., 10., 12., 14.], dtype=float32)
    """


def sub(a: ArrayLike1D, b: ArrayLike1D) -> NDArrayF32:
    """Elementwise subtraction of two 1-D float32 vectors (``a - b``)."""


def scale(a: ArrayLike1D, k: float) -> NDArrayF32:
    """Scale a 1-D float32 vector by scalar ``k`` (``a * k``)."""


def dot(a: ArrayLike1D, b: ArrayLike1D) -> float:
    """Dot product of two 1-D float32 vectors."""


def norm_l2(a: ArrayLike1D) -> float:
    """L2 norm of a 1-D float32 vector (``sqrt(dot(a, a))``)."""


def matmul(A: ArrayLike2D, B: ArrayLike2D) -> NDArrayF32:
    """Matrix multiplication for 2-D float32 arrays (``A @ B``).

    Parameters
    ----------
    A:
        2-D array-like of shape (m, k)
    B:
        2-D array-like of shape (k, n)

    Returns
    -------
    numpy.ndarray
        Float32 array of shape (m, n)

    Raises
    ------
    ValueError
        If shapes are incompatible or inputs are not 2-D.
    """
