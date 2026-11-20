# 与「编译.md」完全对齐的 CMake 流程
cmake_minimum_required(VERSION 3.16)
project(rvv LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

# 1. 读取 sdkconfig 生成的标志
file(READ compile_flags.txt CXX_FLAGS_RAW)
if(CXX_FLAGS_RAW)
    # 将标志分割为列表并逐个添加
    string(REGEX REPLACE " +" ";" CXX_FLAGS_LIST "${CXX_FLAGS_RAW}")
    foreach(flag ${CXX_FLAGS_LIST})
        # 避免传递可能导致RVV问题的标志
        if(NOT flag MATCHES "-flto")
            add_compile_options(${flag})
        endif()
    endforeach()
endif()

file(STRINGS compile_definitions.txt DEFS_RAW)
if(DEFS_RAW)
    foreach(d ${DEFS_RAW})
        add_definitions(-D${d})
    endforeach()
endif()

# 2. 找 pybind11
find_package(pybind11 REQUIRED)

# 3. 源文件
file(GLOB_RECURSE SOURCES src/*.cpp)

# 4. 生成 Python 扩展模块
pybind11_add_module(rvv ${SOURCES})
set_property(TARGET rvv PROPERTY INTERPROCEDURAL_OPTIMIZATION OFF)

# 5. 安装到 build/dist 方便打包
install(TARGETS rvv DESTINATION build/dist)

# 6. 生成 whl（setuptools）
add_custom_command(TARGET rvv POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Packaging wheel..."
    COMMAND python3 setup.py bdist_wheel
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Building whl..."
)