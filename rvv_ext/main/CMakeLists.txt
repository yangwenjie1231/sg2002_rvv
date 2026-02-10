############### Add include ###################
list(APPEND ADD_INCLUDE "include")

############ Add source files #################
append_srcs_dir(ADD_SRCS "src")
# Force-link core sources into the shared library.
# Some SDK/component scan setups may miss sources collected by aux_source_directory.
list(APPEND ADD_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/bind_rvv.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/rvv_dispatch.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/rvv_scalar.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/rvv_rvv071.cpp"
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Prefer optimization for numeric kernels.
list(APPEND ADD_DEFINITIONS -O3)

# If building for SG2002/C906FDV with the provided musl toolchain, enable RVV 0.7.1 flags.
# These flags match the toolchain metadata you provided:
#   -mcpu=c906fdv -march=rv64imafdcv0p7xthead -mcmodel=medany -mabi=lp64d
# Notes:
# - The SDK sets the compiler via CONFIG_TOOLCHAIN_PATH/CONFIG_TOOLCHAIN_PREFIX.
# - We still need to pass the ISA/ABI flags to actually enable RVV.
if(MaixCam)
    list(APPEND ADD_DEFINITIONS
        -mcpu=c906fdv
        -march=rv64imafdcv0p7xthead
        -mcmodel=medany
        -mabi=lp64d
        -DRVV_SG2002_C906FDV=1
    )
endif()

# Python headers/libs selection.
# - For host Linux builds, try system Python dev first.
# - If Python.h isn't available (common on minimal embedded build hosts), fall back to the SDK-provided
#   python3_lib_maixcam_musl_3.11.6 component headers/libs so pybind11 can compile.
if(Linux)
    find_path(_PYTHON_H_PATH Python.h
        PATHS /usr/include /usr/local/include
        PATH_SUFFIXES python3.11 python3.11m python3.12 python3.12m python3.10 python3.10m python3.9 python3.9m
    )
    if(_PYTHON_H_PATH)
        execute_process(COMMAND python3 -c "import sys; print(sys.executable)" OUTPUT_VARIABLE Python3_EXECUTABLE OUTPUT_STRIP_TRAILING_WHITESPACE)
        find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
        list(APPEND ADD_INCLUDE ${Python3_INCLUDE_DIRS})
        list(APPEND ADD_LINK_SEARCH_PATH ${Python3_LIBRARY_DIRS})
        list(APPEND ADD_REQUIREMENTS ${Python3_LIBRARIES})
    else()
        message(WARNING "Python.h not found for host Linux build; falling back to SDK python3_lib_maixcam_musl_3.11.6 headers/libs")
        list(APPEND ADD_REQUIREMENTS python3_lib_maixcam_musl_3.11.6)
    endif()
elseif(MaixCam)
    list(APPEND ADD_REQUIREMENTS python3_lib_maixcam_musl_3.11.6)
elseif(MaixCam2)
    list(APPEND ADD_REQUIREMENTS python3.13_maixcam2)
endif()

list(APPEND ADD_REQUIREMENTS pybind11)

register_component(DYNAMIC)