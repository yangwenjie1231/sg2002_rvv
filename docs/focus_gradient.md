# 对焦梯度统计接入指南

完整 API 契约见 [`focus_gradient_api.md`](focus_gradient_api.md)。本页给
出固件/MaixPy 侧最短接入路径。

## 为什么不是直接返回 score

`rvv.gradient_stats_u8` 只返回：

```python
{
    "grad_sum": float,
    "lum_sum": float,
    "lum_sq_sum": float,
    "count": int,
    "saturated_count": int,
}
```

对焦分数由 Python 侧用与现有 `evaluate_sharpness_diagnostics()` 相同的
公式计算，便于逐项对齐和回归：

```python
mean = stats["lum_sum"] / stats["count"]
var = max(0.0, stats["lum_sq_sum"] / stats["count"] - mean * mean)
std = var ** 0.5
sat_ratio = stats["saturated_count"] / stats["count"]
score = stats["grad_sum"] / (
    stats["count"] * (mean + noise_floor) ** 2
)
```

`noise_floor`、暗场/平坦场闸门仍由上层策略控制。

## 从 MaixPy Image 调用

```python
from maix import camera, image
import rvv

cam = camera.Camera(640, 480)
img = cam.read()

# 只接受 1（灰度）或 3（RGB/BGR）通道；RGBA/BGRA 先转 RGB888。
if img.format() not in (image.Format.FMT_RGB888, image.Format.FMT_BGR888):
    img = img.to_format(image.Format.FMT_RGB888)

raw = img.to_bytes()          # bytes，行紧凑；不需要 NumPy
w, h = img.width(), img.height()
stride = len(raw) // h

stats = rvv.gradient_stats_u8(
    raw, w, h, stride,
    roi_fraction=0.7,
    step_x=0,                 # 0 = max(2, roi_w // 100)
    step_y=0,                 # 0 = max(2, roi_h // 100)
    offset=2,                 # 前向差分 offset，不是采样相位
    channels=3,
)
```

也可以直接传 `bytearray`/`memoryview`：

```python
buf = bytearray(w * h * 3)
stats = rvv.gradient_stats_u8(buf, w, h, w * 3, channels=3)
```

## 参数速查

| 参数 | 固件默认 | 说明 |
| --- | --- | --- |
| `roi_fraction` | `0.7` | 中心 ROI 比例，内部钳制到 `[0.1, 1.0]` |
| `step_x/step_y` | `0` | 自动按 ROI `//100`，最小 2 |
| `offset` | `2` | 前向差分像素偏移，必须 >=1 |
| `channels` | `3` | `1`=灰度，`3`=RGB/BGR |

ROI 与采样公式：

```text
roi_w = max(4, int(width  * roi_fraction))
roi_h = max(4, int(height * roi_fraction))
x0 = (width - roi_w) // 2
y0 = (height - roi_h) // 2
step_x = max(2, roi_w // 100)  # 仅 step_x == 0
step_y = max(2, roi_h // 100)  # 仅 step_y == 0
for y in range(y0, y0 + roi_h - offset, step_y):
    for x in range(x0, x0 + roi_w - offset, step_x):
```

## 兼容性注意

- RGB 灰度是 `(R + G + B) / 3.0`，**不是 Rec.601**；因为三通道等权，
  RGB888 和 BGR888 结果一致。
- `saturated_count` 按灰度 `c >= 250.0` 计数，不是任一通道。
- 边界不足 `offset` 的采样点跳过；`width/height < 4` 返回全零。
- 输出 dict 不依赖 NumPy；`abs_f32/sum_f32/...` 等旧接口仍返回
  ndarray，需要 NumPy 环境。
- RVV 路径用 float32 向量累加，scalar 路径用 double；统计整数字段应
  完全一致，浮点字段用相对误差比较。
- 目标机 RVV 分支尚未在本机交叉编译/实机验证，需由环境负责人补跑。
