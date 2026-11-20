# `gradient_stats_u8` API 契约（SG2002 / C906 RVV 0.7.1）

状态：**契约文档**。当前实现见 `src/image_stats.cpp` 与 `src/pybind_rvv.cpp`；本文只描述公开行为，不改 `src/`、不构建。

目标：用一次原始字节扫描替代现有 Python `evaluate_sharpness_diagnostics()` 的 `get_pixel()` 循环；返回对焦诊断所需的原始统计量，再由 Python 侧组合出与现有实现一致的 `score`。不依赖 NumPy。

---

## 1. 签名与返回

```python
def gradient_stats_u8(
    buf,
    width: int,
    height: int,
    stride: int,
    roi_fraction: float = 0.7,
    step_x: int = 0,
    step_y: int = 0,
    offset: int = 2,
    channels: int = 3,
) -> dict:
    ...
```

`stride` 是**必填的位置参数**，不是关键字参数。当前 pybind11 绑定未加 `kw_only`，所以后面的参数也允许按位置传递；生产代码建议全部用关键字，便于阅读和升级。

返回 dict 只含 5 个原始统计量：

```python
{
    "grad_sum": float,          # Σ(dx² + dy²)
    "lum_sum": float,           # Σ 灰度
    "lum_sq_sum": float,        # Σ 灰度²
    "count": int,               # 有效采样点数
    "saturated_count": int,     # 灰度 >= 250 的采样点数
}
```

它**不返回** `score`、`mean_lum`、`std_lum`、`saturated_ratio`、`roi_fraction`、`legacy_score`、`metric`。这些是 `evaluate_sharpness_diagnostics()` 的展示/门限字段，由 Python 适配器计算（见第 6 节）。这样划分是为了让 C++ 扩展只负责像素扫描，不把 firmware 的配置常量固化进二进制。

---

## 2. 参数

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `buf` | buffer protocol | 必填 | C-contiguous、8-bit 像素数据。`bytes`、`bytearray`、`memoryview`（1-D/2-D/3-D 均可）等；不接受 list/tuple/str。shape 只用于连续性校验，实际按 `stride`/`channels` 寻址。 |
| `width` | `int` | 必填 | 图像宽度（像素），`>= 0`；有效计算要求 `>= 4`。 |
| `height` | `int` | 必填 | 图像高度（像素），`>= 0`；有效计算要求 `>= 4`。 |
| `stride` | `int` | 必填 | 相邻两行起始地址的**字节**距离，必须 `>= width * channels`。 |
| `roi_fraction` | `float` | `0.7` | 中心 ROI 边长比例；先钳制到 `[0.1, 1.0]`。`NaN`/`inf` 报错。 |
| `step_x` | `int` | `0` | 水平采样步长；`0` 表示按 ROI 宽度自动计算。显式值必须 `>= 1`。 |
| `step_y` | `int` | `0` | 垂直采样步长；`0` 表示按 ROI 高度自动计算。显式值必须 `>= 1`。 |
| `offset` | `int` | `2` | 前向差分偏移（像素），必须 `>= 1`。对焦兼容模式固定用默认 `2`。 |
| `channels` | `int` | `3` | `3` = `RGB888`（每像素 3 字节，`R,G,B`）；`1` = `GRAY8`（每像素 1 字节）。 |

`bpp = channels`。

---

## 3. 输入 buffer 契约

### 3.1 支持的格式

**RGB888（`channels=3`，默认）**

```text
row 0: R0 G0 B0 R1 G1 B1 ... R(w-1) G(w-1) B(w-1) [row padding]
row 1: ...
```

- 每像素 3 字节，通常为 `R,G,B`。
- 因灰度是 `(R + G + B) / 3.0` 等权平均，`BGR888` 与 `RGB888` 的统计结果相同；`RGBA/BGRA`、YUV、Bayer、JPEG/PNG 压缩数据不支持。
- 不做色彩空间转换；不是 BT.601/BT.709 加权灰度。

**GRAY8（`channels=1`）**

```text
row 0: Y0 Y1 ... Y(w-1) [row padding]
row 1: ...
```

- 每像素 1 字节，取值 `0..255`。

### 3.2 C-contiguous 与 stride

- `buf` 必须是 C-contiguous buffer，`itemsize == 1`，PEP 3118 format 为无符号字节 `B`（实现也接受 `c`）。1-D/2-D/3-D shape 均可，shape 不参与 ROI 计算。非连续切片、负 stride、转置视图会被拒绝。
- `stride` 是**行 pitch（字节）**，不是像素数：

```text
row_bytes = width * channels
stride >= row_bytes
```

- `stride > row_bytes` 表示行尾有 padding；每行只读前 `row_bytes` 字节，跳过 padding。列方向仍固定为 `channels` 字节，不支持任意二维 stride。
- 最小 buffer 长度：

```text
required = stride * (height - 1) + row_bytes      # height >= 1
```

`len(buf) >= required`；尾部多余字节允许存在并忽略；不足报 `ValueError`。
- ROI 不从 0 开始时，仍用完整图像的 `stride` 定位行。

### 3.3 MaixPy `Image` 衔接

`maix.image.Image.to_bytes(copy=True)` 返回原始数据（`api_maix_image.md` 的 `to_bytes` 条目）：

```python
raw = img.to_bytes(copy=True)     # 推荐：拥有独立生命周期
```

`copy=False` 的 bytes 在 Image 释放后会失效，可能导致崩溃；除非调用期间严格保持 Image 存活，否则不要用。

MaixPy 没有公开的行 pitch 属性：

- 若 `len(raw) == height * width * channels`，直接传 `stride = width * channels`；
- 若 `len(raw) > height * width * channels` 且能被 `height` 整除，才可尝试 `stride = len(raw) // height`；
- 不能整除时不要猜，先确认相机/格式的实际布局。

---

## 4. ROI、step、offset 语义

### 4.1 中心 ROI

完全复刻现有 Python 实现：

```python
f = min(max(float(roi_fraction), 0.1), 1.0)

roi_w = max(4, int(width * f))
roi_h = max(4, int(height * f))

x0 = (width - roi_w) // 2
y0 = (height - roi_h) // 2
```

- ROI 为半开区间 `[x0, x0+roi_w) × [y0, y0+roi_h)`。
- `int()` 是向零截断，不是四舍五入；`max(4, ...)` 是现有实现的兜底。对常见的 `width,height >= 8` 和 `f >= 0.1`，它不改变 `int(width*f)` 的结果；仅在极小图（如 `width=4`）时把 ROI 兜到 4。
- `width < 4` 或 `height < 4`：返回全零统计（`count=0`），不采样。
- 0 字节 buffer：返回全零统计（即使传入了非零的 width/height），不会越界。
- `f` 只在 `[0.1, 1.0]` 内取值；传入 `0.01` 按 `0.1` 处理，传入 `2.0` 按 `1.0` 处理。
- 该函数不返回 `roi_fraction`；适配器使用的就是这里钳制后的 `f`（不是 `roi_w / width`）。

### 4.2 采样步长

```python
if step_x == 0:
    step_x = max(2, roi_w // 100)
if step_y == 0:
    step_y = max(2, roi_h // 100)
```

- 默认（`0`）约采 100×100 个点；ROI 很小时退化为步长 2。
- `step_x`/`step_y` 显式为正整数时直接使用，便于测试和特殊场景；**对焦兼容模式必须用默认 0**，否则分数与现有曲线不可比。
- 采样从 ROI 左上角 `(x0,y0)` 开始，按行优先，不额外居中。

### 4.3 前向差分

`offset` 默认 2：

```python
for y in range(y0, y0 + roi_h - offset, step_y):
    for x in range(x0, x0 + roi_w - offset, step_x):
        c  = L(x,         y)
        cx = L(x + offset, y)
        cy = L(x,         y + offset)

        dx = cx - c
        dy = cy - c
        grad_sum += dx*dx + dy*dy
        lum_sum += c
        lum_sq_sum += c*c
        if c >= 250.0:
            saturated_count += 1
        count += 1
```

- 一阶**前向差分**，不是中心差分；不做 `dx/2`、`dy/2` 归一化。
- `x+offset`、`y+offset` 必须仍在 ROI 内；循环上界已排除边界。
- 不做边界复制（replicate padding）。
- `offset >= roi_w` 或 `offset >= roi_h` 时无有效采样点，返回全零统计（实现中的行为）。
- 采样点数：

```text
count_x = len(range(0, roi_w - offset, step_x))
count_y = len(range(0, roi_h - offset, step_y))
count   = count_x * count_y
```

---

## 5. 统计量定义

对每个采样点，令 `c` 为灰度：

```text
grad_sum   = Σ (dx² + dy²)
lum_sum    = Σ c
lum_sq_sum = Σ c²
count      = 采样点数
saturated_count = count(c >= 250)
```

其中：

```text
dx = L(x + offset, y) - L(x, y)
dy = L(x, y + offset) - L(x, y)
```

`L` 的定义：

- `channels=3`：`L = (R + G + B) / 3.0`；
- `channels=1`：`L = gray_byte`。

输出为 Python `float`（C++ `double` 转换）；`count`、`saturated_count` 为 Python `int`。

实现可以用整数累加避免逐像素浮点：

- RGB：令 `s = R+G+B`，`dx_num = s(x+offset,y) - s(x,y)`，`dy_num = s(x,y+offset) - s(x,y)`；
  `grad_sum = (Σ(dx_num² + dy_num²)) / 9.0`。
- `lum_sum = Σ s / 3.0`，`lum_sq_sum = Σ s² / 9.0`。
- GRAY8 直接累加 `g`、`g²`、`dg²`。

当前实现使用 `double` 标量路径；RVV 路径使用 float32 向量累加，因此与纯 Python 参考值可能有很小误差，测试应按相对误差比较（现有测试用 `2e-6`）。

---

## 6. 与 `evaluate_sharpness_diagnostics` 的对应关系

现有实现：`D:\projects\Cell-Vision-Pro-v2\firmware\utils\image_utils.py`（约 163–294 行）；配置常量：`firmware/config.py`（约 81–87 行）：

```python
FOCUS_ROI_FRACTION = 0.7
FOCUS_MIN_MEAN_LUM = 8.0
FOCUS_MIN_STD_LUM = 2.0
FOCUS_NOISE_FLOOR = 16.0
```

### 6.1 派生字段

```python
def diagnostics_from_stats(stats, roi_fraction):
    count = stats["count"]
    out = {
        "score": 0.0,
        "mean_lum": 0.0,
        "std_lum": 0.0,
        "grad_sum": float(stats["grad_sum"]),
        "count": count,
        "saturated_ratio": 0.0,
        "roi_fraction": 0.0,
        "legacy_score": 0.0,
        "metric": "normalized_gradient_v2",
    }
    if count <= 0:
        return out

    mean_lum = stats["lum_sum"] / count
    variance = max(0.0, stats["lum_sq_sum"] / count - mean_lum * mean_lum)
    std_lum = variance ** 0.5
    saturated_ratio = stats["saturated_count"] / count

    out.update({
        "mean_lum": mean_lum,
        "std_lum": std_lum,
        "saturated_ratio": saturated_ratio,
        "roi_fraction": roi_fraction,
    })

    # 与现有实现一致：门限不过时，legacy_score 也保持 0
    if mean_lum < 8.0:
        return out
    if std_lum < 2.0:
        return out

    out["legacy_score"] = stats["grad_sum"] / (count * mean_lum * mean_lum)
    out["score"] = stats["grad_sum"] / (
        count * (mean_lum + 16.0) ** 2
    )
    return out
```

### 6.2 字段对应表

| 现有 `evaluate_sharpness_diagnostics` | 来源 |
|---|---|
| `mean_lum` | `lum_sum / count` |
| `std_lum` | `sqrt(max(0, lum_sq_sum/count - mean_lum²))` |
| `grad_sum` | 直接来自扩展 |
| `count` | 直接来自扩展 |
| `saturated_ratio` | `saturated_count / count` |
| `roi_fraction` | `clamp(FOCUS_ROI_FRACTION, 0.1, 1.0)` |
| `legacy_score` | 通过门限后 `grad_sum/(count*mean_lum²)` |
| `score` | 通过门限后 `grad_sum/(count*(mean_lum+16)²)` |
| `metric` | 固定 `"normalized_gradient_v2"` |

### 6.3 完整适配器（保持 HTTP 层不抛异常）

```python
from maix import image
import rvv

ZERO = {
    "score": 0.0,
    "mean_lum": 0.0,
    "std_lum": 0.0,
    "grad_sum": 0.0,
    "count": 0,
    "saturated_ratio": 0.0,
    "roi_fraction": 0.0,
    "legacy_score": 0.0,
    "metric": "normalized_gradient_v2",
}


def evaluate_sharpness_diagnostics_fast(img, roi_fraction=0.7):
    out = dict(ZERO)
    try:
        w, h = img.width(), img.height()
        if w < 4 or h < 4:
            return out

        fmt = img.format()
        if fmt == image.Format.FMT_RGB888:
            channels = 3
        elif fmt == image.Format.FMT_GRAYSCALE:
            channels = 1
        else:
            img = img.to_format(image.Format.FMT_RGB888)
            w, h = img.width(), img.height()
            channels = 3

        raw = img.to_bytes(copy=True)
        stride = w * channels
        if len(raw) < stride * h:
            raise ValueError("image buffer smaller than width*height*channels")

        stats = rvv.gradient_stats_u8(
            raw, w, h, stride,
            roi_fraction=roi_fraction,
            channels=channels,
        )
        return diagnostics_from_stats(stats, min(max(float(roi_fraction), 0.1), 1.0))
    except Exception:
        import logging
        logging.getLogger(__name__).exception("gradient_stats_u8 failed")
        return out
```

注意：

- `diagnostics_from_stats` 使用第 6.1 节的实现；生产代码应把 `0.7/8.0/2.0/16.0` 从 `config.py` 读取，不要重复硬编码。
- `stride = w * channels` 只适用于紧密排列；若相机有行 padding，按第 3.3 节确认真实 pitch。
- 核心 `rvv.gradient_stats_u8` 对非法 buffer 抛异常；HTTP 路由需要保留旧的“失败返回零 dict”行为，所以适配器捕获并记录日志。

### 6.4 重要风险：现有 `get_pixel` 路径可能返回 `count=0`

现有 Python 函数把 `img.get_pixel(x, y)` 的返回值当作 `int` 或至少 3 元素的 tuple/list。但 MaixPy v4 文档和 MaixCDK 头文件表明：

- `FMT_RGB888` 默认 `rgbtuple=False`：返回单元素 list `[packed_RGB]`；
- `FMT_GRAYSCALE`：返回单元素 list `[gray]`。

现有代码：

```python
if isinstance(p, (list, tuple)):
    c = (p[0] + p[1] + p[2]) / 3.0
```

对单元素 list 取 `p[1]` 会触发 `IndexError`，随后被 `except Exception: continue` 吞掉，最终可能 `count == 0`、`score == 0`。现有测试的 `FakeImage` 返回 `int` 或三元组，覆盖不到真实 `maix.image.Image`。

**结论**：`gradient_stats_u8` 直接扫描 `to_bytes()` 的原始字节，既避免 `get_pixel` 的返回形态歧义，也避免逐像素 Python 调用。该风险需在真机上分别用 RGB888、GRAYSCALE 复现确认；即使绑定层实际做了特殊处理，raw-byte 路径仍是更稳的接口。

---

## 7. Python 示例（不使用 NumPy）

### 7.1 纯 bytes 确定性例子

构造 8×6 灰度图，`gray(x,y) = 10*y + x`；取 `roi_fraction=1.0`，默认 `step=2`、`offset=2`。

```python
import rvv

w, h = 8, 6
buf = bytes(10 * y + x for y in range(h) for x in range(w))

stats = rvv.gradient_stats_u8(
    buf, w, h, w,        # stride = w（GRAY8 紧密排列）
    roi_fraction=1.0,
    channels=1,
)

print(stats)
```

期望：

```python
{
    "grad_sum": 2424.0,      # 6 个采样点，每点 dx=2, dy=20
    "lum_sum": 72.0,         # 0+2+4+20+22+24
    "lum_sq_sum": 1480.0,    # 0²+2²+4²+20²+22²+24²
    "count": 6,
    "saturated_count": 0,
}
```

采样点为 `(0,0),(2,0),(4,0),(0,2),(2,2),(4,2)`；`count_x=3`、`count_y=2`。由这些统计量可得：

```python
mean_lum = 72.0 / 6          # 12.0
std_lum = (1480.0 / 6 - 12.0 ** 2) ** 0.5   # 10.132456102380441
score = 2424.0 / (6 * (12.0 + 16.0) ** 2)   # 0.5153061224489796
```

### 7.2 MaixPy 相机 RGB888

```python
from maix import camera, image
import rvv

cam = camera.Camera(640, 480, image.Format.FMT_RGB888)
frame = cam.read()

raw = frame.to_bytes(copy=True)
w, h = frame.width(), frame.height()
stride = w * 3
assert len(raw) >= stride * h

stats = rvv.gradient_stats_u8(
    raw, w, h, stride,
    roi_fraction=0.7,
    channels=3,
)

print("grad_sum =", stats["grad_sum"])
print("count =", stats["count"])
print("saturated =", stats["saturated_count"])
```

相机默认输出 `FMT_RGB888`（`api_maix_camera.md` 的 `Camera.__init__`）。若相机输出 `FMT_GRAYSCALE`，用 `stride = w`、`channels=1`。

---

## 8. 错误行为

当前实现的主要校验（均应在扫描前完成）：

| 情况 | 异常 / 结果 |
|---|---|
| `buf` 不是 buffer protocol 对象 | `TypeError`（pybind11 参数转换失败） |
| `buf` 非连续 / 非 1-byte itemsize / 非 `B` 或 `c` | `ValueError`，消息以 `[gradient_stats_u8]` 开头 |
| `width`/`height`/`stride`/`step_x`/`step_y`/`offset` 为负 | `ValueError` |
| `offset < 1` | `ValueError` |
| `channels` 不是 1 或 3 | `ValueError` |
| `roi_fraction` 为 `NaN`/`inf` | `ValueError` |
| `stride < width * channels` | `ValueError`，报出两者 |
| buffer 长度不足 | `ValueError`，报出 `required` 和实际字节数 |
| 0 字节 buffer | 返回全零统计，不抛异常 |
| `width < 4` 或 `height < 4` | 返回全零统计，不抛异常 |
| `offset >= roi_w` 或 `offset >= roi_h` | 返回全零统计（无有效采样点） |
| 全黑/平坦图 | 合法统计：`count > 0`，`grad_sum=0`，`saturated_count=0` |

核心函数不应吞掉异常返回“假零”；只有“图像太小/无采样点”这类**合法空结果**才返回全零。firmware 的 HTTP 层若需要“永不抛异常”，用第 6.3 节的适配器捕获。

线程/生命周期：

- 函数只读 buffer；调用期间不要由其他线程 resize/reallocate 同一块内存。
- 若实现释放 GIL，调用方必须保证 buffer 存活；`to_bytes(copy=True)` 最简单。

---

## 9. 测试策略

### 9.1 主机测试（无需 MaixPy / NumPy）

当前已有 `tests/test_gradient_stats.py`，覆盖：

- 61×47 灰度随机图案与 Python 参考实现对比；
- RGB 与灰度等价性（三通道同值）；
- 64×48 RGB 彩色图案、显式 `step_x=3, step_y=2, offset=3`；
- `stride = width + 7` 的 padding buffer；
- `roi_fraction` 钳制（0、负数、1.0）；
- 0×0、3×8、8×3 小图；
- `offset` 超过 ROI；
- 全 255 饱和图；
- 非法 `channels`、负步长/负 offset、stride 太小、buffer 不足；
- 非连续 `memoryview` 拒绝。

建议补充：

1. 8×6 golden 例子（第 7.1 节）的精确字段断言。
2. RGB 平坦图 `(R,G,B)=(10,20,30)`：`lum_sum = 20*N`，`grad_sum=0`。
3. 单行/单列斜坡，验证 `dx`、`dy` 的符号和 offset=2。
4. `roi_fraction` 边界 `w=4,h=4` 的 `max(4,...)` 行为。
5. 与现有 `evaluate_sharpness_diagnostics` 的 Python 参考实现做随机差分（用修正后的 `get_pixel(..., rgbtuple=True)` 或 raw bytes）。
6. 大图（640×480）性能与数值稳定性：RVV float32 累加与 double 参考的相对误差，建议阈值 `1e-5`～`1e-4`，不要用严格相等。

### 9.2 真机

- 用 MaixCAM 工具链编译，确认 `__riscv_vector` 分支生效（见 `build_notes.md`）。
- 640×480 RGB888 / GRAYSCALE 各跑 100 次：
  - 对比 Python 参考实现的 `grad_sum`、`lum_sum`、`lum_sq_sum`、`count`、`saturated_count`；
  - 记录 `to_bytes(copy=True)` 拷贝 + 内核的端到端时间，不只报 kernel 时间；
  - 用 `objdump` 确认有 `vsetvli` 等向量指令；对比标量构建，确认不是伪加速。
- 打印 `type(img.get_pixel(0,0))`，验证第 6.4 节的风险是否在真机出现。

---

## 10. RVV 0.7.1 实现依据

- `vlse8_v_u8m*` 支持按字节跨步加载（手册 §7.2，PDF p.55；原型 p.59），适合按 `step_x*channels` 取同一行的采样点；RGB 三通道可用 `bstride=3`。
- `vwaddu` / `vwcvtu` / `vfwcvt` 可做 u8→u16→f32 扩展（p.571）。
- 平方/乘加可用 `vwmulu`（p.436）、`vwmaccu`（p.464）。
- 求和可用 `vredsum`（p.579–580）、`vwredsumu`（p.597）。
- 手册 §18（p.19）的 `vdot` intrinsic 列表仍标 TODO；不要依赖 `vdot`，用 widening multiply + reduction。
- 饱和计数可用 `vmfge`/`vmerge` 或比较掩码 + `vpopc`（p.608）。
- 0.7.1 与 1.0 的命名差异见手册 §20（p.19–24），构建约束见 `build_notes.md`。

---

## 11. 参考

- 现有算法与字段：`D:\projects\Cell-Vision-Pro-v2\firmware\utils\image_utils.py`，`evaluate_sharpness_diagnostics`，约 163–294 行。
- 配置常量：`D:\projects\Cell-Vision-Pro-v2\firmware\config.py`，约 81–87 行。
- 现有测试：`D:\projects\Cell-Vision-Pro-v2\firmware\tests\test_focus_sharpness.py`。
- 当前扩展实现：`src/image_stats.cpp`、`src/pybind_rvv.cpp`、`tests/test_gradient_stats.py`。
- MaixPy 图像 API：`D:\projects\doc\maixpy-docs-flat\maixpy-docs-flat\api_maix_image.md`
  - `Format`：第 16–57 行；
  - `data_size`：第 5863–5869 行；
  - `get_pixel`：第 5958–5978 行；
  - `to_bytes`：第 6064–6080 行；
  - `to_format`：第 6089–6103 行。
- MaixPy 相机 API：`D:\projects\doc\maixpy-docs-flat\maixpy-docs-flat\api_maix_camera.md`
  - `Camera.__init__` 默认 `FMT_RGB888`：第 146–170 行；
  - `read`：第 219–241 行。
- MaixCDK `get_pixel` 返回 `std::vector<uint32_t>`：
  `https://github.com/sipeed/MaixCDK/blob/main/components/vision/include/maix_image.hpp`
- RVV 0.7.1 intrinsic 手册：
  `D:\projects\rvv-ext\Xuantie+900+Series+RVV-0.7.1+Intrinsic+Manual.pdf`
  - §7.2 / p.55、p.59；p.436；p.464；p.571；p.579–580；p.597；§18 / p.19；§20 / p.19–24。
