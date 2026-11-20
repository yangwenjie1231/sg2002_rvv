# rvv API 参考

除 `gradient_stats_u8` 外，函数均使用 NumPy `ndarray` 输入/输出；
`gradient_stats_u8` 接受 PEP 3118 buffer 并返回 Python `dict`。

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

## 通用 float32 扩展
- `rvv.abs_f32(a)` → ndarray
- `rvv.sum_f32(a)` → float（内部 double 累加）
- `rvv.mean_std_f32(a)` → `(mean, population_std)`
- `rvv.mul_f32(a, b)` → ndarray（等长校验）
- `rvv.add_scalar_f32(a, k)` → ndarray
- `rvv.clip_f32(a, lo, hi)` → ndarray（保留 NaN，`lo > hi` 报错）
- `rvv.max_f32(a)` → float（空数组/含 NaN 返回 NaN）
- `rvv.min_f32(a)` → float（空数组/含 NaN 返回 NaN）

## 图像对焦统计
- `rvv.gradient_stats_u8(buf, width, height, stride, roi_fraction=0.7, step_x=0, step_y=0, offset=2, channels=3)` → dict
  输入任意 PEP 3118 uint8 C-contiguous buffer；返回 `grad_sum`、`lum_sum`、
  `lum_sq_sum`、`count`、`saturated_count`。详见
  [focus_gradient_api.md](focus_gradient_api.md)。

## 示例
```python
import numpy as np, rvv
a = np.ones(1024, np.float32)
b = rvv.scale(a, 3.0)
```