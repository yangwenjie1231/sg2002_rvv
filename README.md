# rvv – SG2002 RVV 0.7.1 加速库

## 功能
- 向量级：add / sub / scale / dot / norm_l2 / normalize  
- 矩阵级：add2d / scale2d / matmul / transpose / mv（矩阵×向量）  
- 通用扩展：abs_f32 / sum_f32 / mean_std_f32 / mul_f32 / add_scalar_f32 / clip_f32 / max_f32 / min_f32
- 对焦统计：gradient_stats_u8（uint8 buffer → dict，不要求 NumPy）
- wheel 不强制依赖 NumPy；需要 ndarray 接口时安装 `rvv[numpy]` 或自行安装 NumPy
- 原有 ndarray 接口保持 NumPy 兼容；`gradient_stats_u8` 返回 Python dict
- 内部自动使用玄铁 C906 RVV intrinsics
- **0.2.2 起已在 MaixCAM 真机验证**（C906 / RVV 0.7.1）；`gradient_stats_u8`
  在 640×480 RGB 上约 **3.5 ms**，比纯 Python 采样循环快约 **280×**

对焦接入见 [docs/focus_gradient.md](docs/focus_gradient.md)，完整 API 见
[docs/focus_gradient_api.md](docs/focus_gradient_api.md) 与
[docs/api_extensions.md](docs/api_extensions.md)。

> ⚠️ 写 RVV 代码前必读 [docs/rvv071-device-semantics.md](docs/rvv071-device-semantics.md)：
> 真机的 tail lane 是 agnostic、`vmf*_vf` 与 `vfirst_m` 语义与手册不符，
> 这两类问题在 x86 标量测试和交叉编译里都发现不了。

## 快速安装
```bash
# 在仓库根目录
python project.py build
pip install --no-deps --force-reinstall build/dist/*.whl
```

如果你已经有目标设备可用的 wheel，也可以只安装 wheel：

```bash
pip install --no-deps --force-reinstall rvv-0.2.2-*.whl
```

## 用法示例
```python
import numpy as np
import rvv

a = np.random.rand(1024).astype(np.float32)
b = np.random.rand(1024).astype(np.float32)

c = rvv.add(a, b)        # 向量加法
s = rvv.dot(a, b)        # 点积
n = rvv.norm_l2(a)       # L2 范数
```

## 运行测试
```bash
python tests/test_rvv.py
```

## 文档
详见 `docs/` 目录，或直接在 Python 内 `help(rvv.add)` 查看 docstring。