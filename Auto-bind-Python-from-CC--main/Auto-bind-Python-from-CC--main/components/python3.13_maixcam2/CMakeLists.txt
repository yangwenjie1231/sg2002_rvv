# 只在 MaixCam2 平台编译此组件
if(NOT MaixCam2)
    return()
endif()

# 定义库文件根目录
set(maixcam2_dir "${CMAKE_CURRENT_SOURCE_DIR}/lib")

# 1. 配置头文件路径
list(APPEND ADD_INCLUDE 
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/python3.13"
)

# 2. 添加库搜索路径
list(APPEND ADD_LINK_SEARCH_PATH "${maixcam2_dir}")

# 3. 直接指定 Python 动态库（使用实际文件，而非符号链接）
list(APPEND ADD_DYNAMIC_LIB 
    "${maixcam2_dir}/libpython3.13.so.1.0"
)

# 4. 注册组件
register_component()