#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键构建脚本，完全复用编译.md 模板
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path

CFG_FILE = "sdkconfig"

def run(cmd, check=True):
    print("$", cmd)
    return subprocess.run(cmd, shell=True, check=check)

def menuconfig():
    """启动 kconfig 图形界面"""
    if not Path("Kconfig").exists():
        print("请把 Kconfig 模板放到当前目录")
        sys.exit(1)
    run(f"python3 -m menuconfig Kconfig")

def build():
    """按 sdkconfig 生成 whl"""
    if not Path(CFG_FILE).exists():
        print("sdkconfig 不存在，先执行 menuconfig")
        sys.exit(1)
    # 1. 由 sdkconfig 生成 compile_flags.txt 与 compile_definitions.txt
    run(f"python3 tools/parse_sdkconfig.py {CFG_FILE}")
    # 2. 调用 pybind11 绑定代码生成器
    run("python3 tools/pybind_gen.py")
    # 3. cmake 配置（默认用 Ninja，可改）
    run("cmake -B build -GNinja -DCMAKE_TOOLCHAIN_FILE=tools/toolchain.cmake")
    # 4. 编译 + 打包
    run("cmake --build build")
    print("whl 包已输出到 build/dist/，可直接 pip install")

def clean():
    if Path("build").exists():
        shutil.rmtree("build")
    for p in Path(".").rglob("*.pyc"):
        p.unlink()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python project.py [menuconfig|build|clean]")
        sys.exit(0)
    act = sys.argv[1]
    if act == "menuconfig":
        menuconfig()
    elif act == "build":
        build()
    elif act == "clean":
        clean()
    else:
        print("unknown action:", act)