# rvv – SG2002 RVV 0.7.1 加速库

## 功能
- 向量级：add / sub / scale / dot / norm_l2 / normalize
- 矩阵级：add2d / scale2d / matmul / transpose / mv（矩阵×向量）
- int8：add_i8 / scale_i8 / dot_i8 / add2d_i8 / scale2d_i8
- 通用 float32 扩展：abs_f32 / sum_f32 / mean_std_f32 / mul_f32 / add_scalar_f32 / clip_f32 / max_f32 / min_f32
- wheel 不强制依赖 NumPy；需要 ndarray 接口时安装 `rvv[numpy]` 或自行安装 NumPy
- 内部自动使用玄铁 C906 RVV intrinsics
- **已在 MaixCAM 真机验证**（C906 / RVV 0.7.1）：算子回归 223/223 通过

本库只负责通用 RVV 向量/矩阵加速。图像处理、对焦评分等领域接口由使用方
在自己的工程中实现，不包含在本库内。

完整 API 见 [docs/api.md](docs/api.md) 与
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
pip install --no-deps --force-reinstall rvv-0.3.0-*.whl
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
