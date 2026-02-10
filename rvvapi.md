# rvv API 文档（SG2002 / C906FDV / RVV 0.7.1）

本文档描述 `rvv` Python 扩展模块的公共 API。

- 模块名：`rvv`
- 目标平台：RISC-V（SG2002 / C906FDV），支持 RVV 0.7.1 时会启用矢量加速
- 数据类型：**仅支持 float32**
- 设计目标：
  - **兼容 NumPy 的 N 维数组（ndarray）作为输入**
  - 在内部自动转换为适合 RVV/kernel 处理的格式（float32 + C 连续）
  - 在 NumPy 可用时返回 `numpy.ndarray`（pybind11 `py::array_t<float>`）

> 安装提示：wheel 本身不依赖 NumPy（`install_requires=[]`）。
> 但若要传入 `numpy.ndarray`、或运行与 NumPy 的对比测试，需要目标环境安装 NumPy。

---

## 1. 输入数据模型（重要）

`rvv` 的所有函数都基于 **buffer protocol**（PEP 3118）读取输入。

### 1.1 支持的输入类型

以下输入通常都可以：

- `numpy.ndarray(dtype=np.float32)`
- 其他实现了 buffer protocol 的对象（如某些张量库/自定义 buffer）

### 1.2 强制要求

- **dtype 必须是 float32**
  - PEP3118 `format == 'f'`
  - `itemsize == 4`
- **必须是 C-contiguous（行主序连续）**
  - 若为非连续/切片视图（stride 不满足连续布局），将抛出异常

### 1.3 自动转换规则

内部的 `as_f32_buffer()` 会优先尝试 NumPy 路径：

- `py::array_t<float, c_style | forcecast>`

这意味着：

- 如果 NumPy 可用，且你传入 `float64` 或非连续数组，它会复制/转换为连续的 `float32` 再计算。
- 如果 NumPy 不可用，则只能走通用 buffer 路径，此时不满足上述要求会直接报错。

---

## 2. 向量运算 API（1-D）

### 2.1 `rvv.add(a, b) -> ndarray[float32]`

**功能**：逐元素加法。

**参数**：

- `a`：1-D float32 buffer（期望 `ndim=1`）
- `b`：1-D float32 buffer（期望 `ndim=1`）

**返回**：

- `out`：1-D `float32` 数组，长度与 `a` 相同

**错误**：

- `ValueError`：
  - `a` 或 `b` 维度不是 1：`"a must have ndim=1, got ndim=..."`
  - `a` 与 `b` 长度不同：`"add: a and b must have the same length"`
  - 输入不是 C-contiguous：`"a must be C-contiguous float32 buffer"`
- `TypeError`：
  - 输入不是 float32：例如 `format!='f'` 或 `itemsize!=4`

**示例**：

```python
import numpy as np
import rvv

a = np.arange(8, dtype=np.float32)
b = np.ones(8, dtype=np.float32)
print(rvv.add(a, b))
```

---

### 2.2 `rvv.sub(a, b) -> ndarray[float32]`

**功能**：逐元素减法。

- 语义：`out[i] = a[i] - b[i]`
- 参数/返回/错误与 `add` 相同，长度必须一致。

**示例**：

```python
import numpy as np
import rvv

a = np.arange(8, dtype=np.float32)
b = np.ones(8, dtype=np.float32)
print(rvv.sub(a, b))
```

---

### 2.3 `rvv.scale(a, k) -> ndarray[float32]`

**功能**：向量按标量缩放。

**参数**：

- `a`：1-D float32 buffer
- `k`：Python `float`

**返回**：

- `out`：1-D float32 数组

**错误**：

- `ValueError` / `TypeError`：同 `add` 的 dtype/contiguous/ndim 校验

**示例**：

```python
import numpy as np
import rvv

a = np.arange(8, dtype=np.float32)
print(rvv.scale(a, 0.5))
```

---

### 2.4 `rvv.dot(a, b) -> float`

**功能**：点积（内积）。

- 语义：`sum_i a[i] * b[i]`

**参数**：

- `a`：1-D float32 buffer
- `b`：1-D float32 buffer

**返回**：

- Python `float`

**错误**：

- `ValueError`：长度不一致：`"dot: a and b must have the same length"`
- dtype/连续性/ndim 错误同上

**示例**：

```python
import numpy as np
import rvv

a = np.arange(8, dtype=np.float32)
print(rvv.dot(a, a))
```

---

### 2.5 `rvv.norm_l2(a) -> float`

**功能**：L2 范数（欧几里得范数）。

- 定义：`sqrt(dot(a, a))`

**参数**：

- `a`：1-D float32 buffer

**返回**：

- Python `float`

**数值误差说明（重要）**：

`dot` 和 `norm` 都是“归约”运算（reduction）。
当向量很大时（例如 1e6 长度），由于不同实现的归约顺序不同（RVV 分块、标量循环、NumPy/BLAS 向量化等），最终结果可能与 NumPy 有**很小的相对误差**。

建议比较时使用相对误差（例如 `1e-4 ~ 1e-3`）而不是严格相等。

**示例**：

```python
import numpy as np
import rvv

a = np.random.randn(1000000).astype(np.float32)
print(rvv.norm_l2(a))
```

---

## 3. 矩阵运算 API（2-D）

### 3.1 `rvv.matmul(A, B) -> ndarray[float32]`

**功能**：矩阵乘法。

- 输入：
  - `A`：形状 `(m, k)`
  - `B`：形状 `(k, n)`
- 输出：
  - `C`：形状 `(m, n)`

**参数**：

- `A`：2-D float32 buffer（`ndim=2`，C-contiguous，行主序）
- `B`：2-D float32 buffer（`ndim=2`，C-contiguous，行主序）

**返回**：

- `C`：2-D float32 数组

**错误**：

- `ValueError`：
  - 维度不是 2：`"A must have ndim=2, got ndim=..."`
  - 形状不匹配：
    - `"matmul: shape mismatch: A is (m,k) but B is (k2,n)"`

**示例**：

```python
import numpy as np
import rvv

A = np.random.randn(2, 3).astype(np.float32)
B = np.random.randn(3, 4).astype(np.float32)
C = rvv.matmul(A, B)
print(C.shape)
```

---

## 4. 性能建议

- 确保输入是 **float32** 且 **C-contiguous**，避免不必要的复制。
- 大数组上，`add/sub/scale` 通常更容易看到 RVV 性能收益。
- `matmul` 是否快于 NumPy 取决于 NumPy 是否链接了高性能 BLAS（嵌入式环境中 NumPy 常常没有或性能有限）。

---

## 5. 常见错误与排查

### 5.1 `... must be float32-compatible buffer ...`

说明输入不是 float32。

解决：

```python
x = x.astype(np.float32, copy=False)
```

### 5.2 `... must be C-contiguous float32 buffer`

说明输入是切片/转置等非连续视图。

解决：

```python
x = np.ascontiguousarray(x, dtype=np.float32)
```

### 5.3 `add/sub/dot: a and b must have the same length`

说明向量长度不一致。

---

## 6. 与实现对应关系（供开发者）

- Python 绑定：`rvv_ext/main/src/bind_rvv.cpp`
- C++ 公共接口声明：`rvv_ext/main/include/rvv.hpp`
- RVV 0.7.1 实现：`rvv_ext/main/src/rvv_rvv071.cpp`（仅在 `__riscv_vector` 下编译）
- 标量实现：`rvv_ext/main/src/rvv_scalar.cpp`

---

## 7. 最小示例（组合使用）

```python
import numpy as np
import rvv

a = np.arange(8, dtype=np.float32)
b = np.arange(8, dtype=np.float32)

c = rvv.add(a, b)
d = rvv.sub(a, b)
e = rvv.scale(a, 2.0)

print(c)
print(d)
print(e)
print(rvv.dot(a, b))
print(rvv.norm_l2(a))

A = np.arange(6, dtype=np.float32).reshape(2, 3)
B = np.arange(12, dtype=np.float32).reshape(3, 4)
print(rvv.matmul(A, B))
```
