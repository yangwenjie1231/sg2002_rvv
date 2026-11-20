# 通用 API 扩展

本次扩展遵循「不改变已有 API」原则：原有 `add/sub/scale/dot/norm_l2/
normalize`、二维运算和 int8 运算的签名、返回类型保持不变。新增接口分为
两类：

1. 对焦图像统计：`gradient_stats_u8`（详见
   [`focus_gradient_api.md`](focus_gradient_api.md)）；
2. 通用 float32 算子：`abs_f32`、`sum_f32`、`mean_std_f32`、
   `mul_f32`、`add_scalar_f32`、`clip_f32`、`max_f32`、`min_f32`。

所有新增 float32 算子都同时提供：

- `__riscv_vector` 下的 RVV 0.7.1 实现；
- 非 RVV 主机的标量实现；
- pybind11 绑定；
- NumPy 对比测试（`tests/test_ext_ops.py`）。

## `abs_f32(a) -> ndarray[float32]`

逐元素绝对值：

```text
out[i] = |a[i]|
```

| 项目 | 说明 |
| --- | --- |
| 输入 | 1-D `float32` C-contiguous 数组 |
| 返回 | 新的 1-D `float32` 数组 |
| 空数组 | 返回空数组 |
| RVV | `vle32_v_f32m4` + `vfabs_v_f32m4` + `vse32_v_f32m4` |
| scalar | `std::fabs` |

```python
import numpy as np
import rvv

a = np.array([-2.0, 0.0, 3.5], dtype=np.float32)
print(rvv.abs_f32(a))   # [2.  0.  3.5]
```

## `sum_f32(a) -> float`

```text
sum(a) = Σ a[i]
```

| 项目 | 说明 |
| --- | --- |
| 输入 | 1-D `float32` C-contiguous 数组 |
| 返回 | Python `float` |
| 空数组 | `0.0` |
| RVV | float32 向量按 4096 元素分块累加、块内 `vfredusum` 归约，再逐块加到 `double` 标量 |
| scalar | `double` 顺序累加 |

```python
print(rvv.sum_f32(np.ones(8, dtype=np.float32)))  # 8.0
```

## `mean_std_f32(a) -> (float, float)`

返回样本均值与**总体标准差**（population standard deviation，分母为 `n`，
不是 `n-1`）：

```text
mean = Σ a[i] / n
var  = Σ (a[i] - mean)² / n
std  = sqrt(max(0, var))
```

| 项目 | 说明 |
| --- | --- |
| 输入 | 1-D `float32` C-contiguous 数组 |
| 返回 | `(mean, std)` Python tuple，两个元素均为 `float` |
| 空数组 | `(nan, nan)` |
| 单元素 | `(a[0], 0.0)` |
| RVV | 两遍扫描：先以 double 求和，再用 float32 向量分块计算平方偏差和并逐块累加到 double |
| scalar | double 累加 |

```python
mean, std = rvv.mean_std_f32(
    np.array([1.0, 2.0, 3.0], dtype=np.float32)
)
print(mean, std)   # 2.0 0.816496580927726
```

> 为避免依赖 C906 上并不保证可用的 64-bit 向量元素（部分 C906 资料/
> 工具链将向量 ELEN 限制为 32），RVV 路径不使用 `vfloat64` intrinsic。
> 实现采用 float32 向量分块求和，每块归约结果再累加到 `double` 标量；
> 输入和输出仍是 float32，和 NumPy 的归约顺序不同，比较时建议使用
> 相对误差 `1e-5 ~ 1e-4`。

## `mul_f32(a, b) -> ndarray[float32]`

逐元素乘法：

```text
out[i] = a[i] * b[i]
```

| 项目 | 说明 |
| --- | --- |
| 输入 | 两个 1-D `float32` C-contiguous 数组，长度必须相同 |
| 返回 | 新的 1-D `float32` 数组 |
| 空数组 | 两个空数组返回空数组 |
| 长度不一致 | `ValueError` |
| RVV | `vle32_v_f32m4` × 2 + `vfmul_vv_f32m4` + `vse32_v_f32m4` |
| scalar | 逐元素 `a[i] * b[i]` |

```python
a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
print(rvv.mul_f32(a, b))   # [4. 10. 18.]
```

## `add_scalar_f32(a, k) -> ndarray[float32]`

逐元素加标量：

```text
out[i] = a[i] + k
```

| 项目 | 说明 |
| --- | --- |
| 输入 | 1-D `float32` C-contiguous 数组 + Python `float` |
| 返回 | 新的 1-D `float32` 数组 |
| 空数组 | 返回空数组 |
| RVV | `vfadd_vf_f32m4` |
| scalar | 逐元素加法 |

```python
print(rvv.add_scalar_f32(a, 0.5))   # [1.5 2.5 3.5]
```

## `clip_f32(a, lo, hi) -> ndarray[float32]`

逐元素截断：

```text
out[i] = min(max(a[i], lo), hi)
```

| 项目 | 说明 |
| --- | --- |
| 输入 | 1-D `float32` C-contiguous；`lo`/`hi` 不能为 NaN，且 `lo <= hi` |
| 返回 | 新的 1-D `float32` 数组 |
| 空数组 | 返回空数组 |
| NaN | 保留 NaN（比较为 false，`vmerge` 两侧都不选中） |
| 边界 | 允许 `-inf` / `+inf` 作为边界 |
| RVV | `vmflt_vf_f32m4_b8` / `vmfgt_vf_f32m4_b8` + `vmerge_vvm_f32m4` |
| scalar | 分支比较 |
| 错误 | `lo > hi` 或任一边界为 NaN 时 `ValueError` |

```python
a = np.array([-10, -1, 0.5, 9], dtype=np.float32)
print(rvv.clip_f32(a, -1.0, 2.0))   # [-1. -1.  0.5  2.]
```

## `max_f32(a) -> float`

| 项目 | 说明 |
| --- | --- |
| 输入 | 1-D `float32` C-contiguous 数组 |
| 返回 | Python `float` |
| 空数组 | `NaN` |
| NaN | 任一元素为 NaN 时返回 `NaN`（与 `np.max` 一致，而不是 RISC-V maxNum 忽略 NaN） |
| RVV | `vmfne_vv_f32m4_b8` + `vfirst_m_b8` 检测 NaN，`vfmax_vv_f32m4` 逐元素归并，最后对 VLMAX 个 lane 做标量归约（不依赖 `vfredmax`） |
| scalar | 逐元素比较，遇到 NaN 立即返回 `NaN` |

```python
print(rvv.max_f32(np.array([-5.0, 2.5, 1.0], dtype=np.float32)))  # 2.5
```

## `min_f32(a) -> float`

语义与 `max_f32` 对称：

| 项目 | 说明 |
| --- | --- |
| 空数组 | `NaN` |
| NaN | 任一元素为 NaN 时返回 `NaN` |
| RVV | `vmfne_vv_f32m4_b8` + `vfirst_m_b8` 检测 NaN，`vfmin_vv_f32m4`，最后对 VLMAX 个 lane 做标量归约（不依赖 `vfredmin`） |
| scalar | 逐元素比较，遇到 NaN 立即返回 `NaN` |

```python
print(rvv.min_f32(np.array([-5.0, 2.5, 1.0], dtype=np.float32)))  # -5.0
```

## 输入与错误约定

新增 float32 算子沿用原有封装：

- 必须是一维 `float32` 数组（PEP 3118 format `'f'`）；
- 必须 C-contiguous；
- 非 1-D、dtype 不符、非连续视图会抛 `ValueError` / `TypeError`；
- 成功时返回 `numpy.ndarray` 或 Python 标量，因而不改变原 API 的
  NumPy 使用方式。

`gradient_stats_u8` 是唯一不要求 NumPy 的接口：输入为任意 PEP 3118
uint8 buffer，输出为 Python `dict`。

类型提示见仓库根目录的 `rvv.pyi`（覆盖全部现有函数与新增函数）；它只用于
IDE / 类型检查，不影响运行时。

## RVV 0.7.1 兼容性核对

实现仅使用 `Xuantie+900+Series+RVV-0.7.1+Intrinsic+Manual.pdf` 中列出的
0.7.1 形式（不是 RVV 1.0 形式），包括：

| 用途 | Intrinsic（手册原型） |
| --- | --- |
| 配置 vl | `vsetvl_e32m4(n)`、`vsetvl_e32m2(n)`、`vsetvl_e8m1(n)` |
| 最大 vl | `vsetvlmax_e32m4()`、`vsetvlmax_e32m2()` |
| 连续 load/store | `vle32_v_f32m4`、`vse32_v_f32m4`、`vle32_v_f32m2`、`vle8_v_u8m1` |
| 跨步 load | `vlse32_v_f32m2`、`vlse8_v_u8m1` |
| 浮点逐元素 | `vfabs_v_f32m4`、`vfadd_vv_f32m4`、`vfadd_vf_f32m4`、`vfsub_vf_f32m4`、`vfmul_vv_f32m4`、`vfmax_vv_f32m4`、`vfmin_vv_f32m4` |
| 浮点比较/选择 | `vmfge_vf_f32m2_b16`、`vmflt_vf_f32m4_b8`、`vmfgt_vf_f32m4_b8`、`vmfne_vv_f32m4_b8`、`vfirst_m_b8`、`vmerge_vvm_f32m4`、`vmerge_vvm_u32m2` |
| 扩展/浮点转换 | `vwaddu_vx_u16m2`、`vfwcvt_f_xu_v_f32m4` |
| 整数累加 | `vadd_vv_u32m2`、`vredsum_vs_u32m2_u32m1`、`vmv_x_s_u32m1_u32` |
| 浮点归约 | `vfredusum_vs_f32m4_f32m1`、`vfredusum_vs_f32m2_f32m1`、`vfmv_f_s_f32m1_f32` |

`max_f32/min_f32` 不对 `vfredmax/vfredmin` 做硬依赖：RVV 路径只做
`vfmax_vv`/`vfmin_vv` 的逐元素归并，再把 VLMAX 个 lane 用标量比较收尾。
这样即使目标工具链没有可用的浮点 max/min reduction intrinsic，也不会
影响 Python API；scalar 路径仍可完整作为回退。NaN 检测使用
`vmfne`（NaN != NaN）+ `vfirst_m`，保持与 `np.max/np.min` 一致的
NaN 传播语义。

注意：本仓库的 `compile_flags.txt` 使用 `-march=rv64gcv0p7`；
`__riscv_vector` 下才编译 RVV 分支。`rvv.cpp`、`image_stats.cpp`、
`f32_ext.cpp` 均遵循这一约定。没有使用 RVV 1.0 才有的
`vfredosum` 变体或 `vsetvlmax` 之外的 1.0 专用转置/分段指令。

## 为什么不增加更多算子

当前扩展累计增加 8 个通用 float32 算子（abs/sum/mean_std/mul/
add_scalar/clip/max/min），都保持一维、逐元素或单值归约的小接口，避免为
数量堆叠矩阵/图像专用 API：

- `threshold_u8` 等图像算子暂不加入，优先用 `gradient_stats_u8` 覆盖
  固件需求；
- `matmul` 等已有接口不做重构，避免改变既有行为；
- 任何需要 RVV 1.0 专用指令的候选实现不加入，保持 0.7.1 兼容。

## 测试

```bash
python tests/test_gradient_stats.py   # 不依赖 NumPy/pytest
python tests/test_ext_ops.py          # 需要 NumPy
python tests/test_rvv.py              # 原有回归
python tests/test_int8.py             # 原有回归
```

本机 x86 + MSVC 已验证 scalar 路径；RVV 分支尚未在 SG2002 上编译，
交叉编译验证由环境负责人完成。
