from setuptools import setup, Extension
from pybind11 import get_cmake_dir
import pybind11
from pathlib import Path

ext_modules = [
    Extension(
        "rvv",
        sorted(map(str, Path("src").glob("*.cpp"))),
        include_dirs=[
            "src",
            pybind11.get_include(),
        ],
        language="c++",
        cppstd=17,
        extra_compile_args=["-O3", "-march=rv64gcv0p7"],
    ),
]

setup(
    name="rvv",
    version="0.3.0",
    author="SG2002-Team",
    description="NumPy-compatible RVV 0.7.1 acceleration for SG2002",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    zip_safe=False,
    python_requires=">=3.8",
    # NumPy is optional: the extension accepts any float32 C-contiguous
    # buffer, so it can be installed on firmware images without NumPy.
    install_requires=[],
    extras_require={"numpy": ["numpy"]},
)
