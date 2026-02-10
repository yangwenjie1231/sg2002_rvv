# 只在 MaixCam 平台编译此组件
if(NOT MaixCam)
    return()
endif()

# Kconfig gate
if(DEFINED CONFIG_COMPONENT_PYTHON3_LIB_MAIXCAM_MUSL_3_11_6 AND NOT CONFIG_COMPONENT_PYTHON3_LIB_MAIXCAM_MUSL_3_11_6)
    return()
endif()

# 定义库文件根目录
set(maixcam_dir "${CMAKE_CURRENT_SOURCE_DIR}/lib")

# 1. 配置头文件路径
list(APPEND ADD_INCLUDE 
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/python3.11"
)

# 2. 添加库搜索路径
list(APPEND ADD_LINK_SEARCH_PATH "${maixcam_dir}")

# 3. 直接指定 Python 动态库
list(APPEND ADD_DYNAMIC_LIB 
    "${maixcam_dir}/libpython3.11.so"
)

# 4. 注册组件
register_component()