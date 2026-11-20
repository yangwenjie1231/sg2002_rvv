# rvv API 参考

所有函数均使用 NumPy `ndarray`（或等价的 float32 C-contiguous buffer）作为输入/输出。

## 向量运算
- `rvv.add(a, b)` → ndarray
- `rvv.sub(a, b)` → ndarray
- `rvv.scale(a, k)` → ndarray
- `rvv.dot(a, b)` → float
- `rvv.norm_l2(a)` → float
- `rvv.normalize(a)` → ndarray

## 矩阵运算
- `rvv.add2d(A, B)` → ndarray
- `rvv.scale2d(A, k)` → ndarray
- `rvv.matmul(A, B)` → ndarray
- `rvv.transpose(A)` → ndarray
- `rvv.mv(A, x)` → ndarray  （矩阵 × 向量）

## int8 运算
- `rvv.add_i8(a, b)` → ndarray
- `rvv.scale_i8(a, k)` → ndarray
- `rvv.dot_i8(a, b)` → int
- `rvv.add2d_i8(A, B)` → ndarray
- `rvv.scale2d_i8(A, k)` → ndarray

## 通用 float32 扩展
- `rvv.abs_f32(a)` → ndarray
- `rvv.sum_f32(a)` → float（内部 double 累加）
- `rvv.mean_std_f32(a)` → `(mean, population_std)`
- `rvv.mul_f32(a, b)` → ndarray（等长校验）
- `rvv.add_scalar_f32(a, k)` → ndarray
- `rvv.clip_f32(a, lo, hi)` → ndarray（保留 NaN，`lo > hi` 报错）
- `rvv.max_f32(a)` → float（空数组/含 NaN 返回 NaN）
- `rvv.min_f32(a)` → float（空数组/含 NaN 返回 NaN）

## 示例
```python
import numpy as np, rvv
a = np.ones(1024, np.float32)
b = rvv.scale(a, 3.0)
```

## 范围说明

本库只提供通用 RVV 向量/矩阵加速。图像处理、对焦评分等领域接口由使用方
在自己的工程中实现，不包含在本库内。