#!/usr/bin/env python3
"""
占位脚本：真正项目里可在这里自动生成 pybind 绑定代码，
本例已手写 pybind_rvv.cpp，所以直接跳过。
"""
import sys, os, shutil

bind_file = 'src/pybind_rvv.cpp'
if os.path.exists(bind_file):
    print(f'pybind_gen: 已手写 {bind_file}，跳过自动生成')
else:
    print('pybind_gen: 未找到手写绑定，后续可在此自动生成')