############### Add include ###################
list(APPEND ADD_INCLUDE "include")
list(APPEND ADD_PRIVATE_INCLUDE "")

############ Add source files #################
append_srcs_dir(ADD_SRCS "src")

# ===================== 新增：启用C++17 =====================
# 设置C++标准为17，强制要求（不回退到低版本）
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 禁用编译器扩展（可选，增强跨平台兼容性）
set(CMAKE_CXX_EXTENSIONS OFF)
# ==========================================================

if(Linux)
    ################# Linux #################
    execute_process(COMMAND python3 -c "import sys; print(sys.executable)" OUTPUT_VARIABLE Python3_EXECUTABLE OUTPUT_STRIP_TRAILING_WHITESPACE)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
    list(APPEND ADD_INCLUDE ${Python3_INCLUDE_DIRS})
    list(APPEND ADD_LINK_SEARCH_PATH ${Python3_LIBRARY_DIRS})

    list(APPEND ADD_REQUIREMENTS ${Python3_LIBRARIES})

elseif(MaixCam)
    ################# MaixCam #################
    list(APPEND ADD_REQUIREMENTS python3_lib_maixcam_musl_3.11.6)

elseif(MaixCam2)
    ################ MaixCam2 #################
    list(APPEND ADD_REQUIREMENTS python3.13_maixcam2)
endif()

###### Add required/dependent components ######
list(APPEND ADD_REQUIREMENTS pybind11)

# if(CONFIG_COMPONENT1_ENABLED)
#     list(APPEND ADD_REQUIREMENTS component2)
# endif()
# if(CONFIG_COMPONENT2_ENABLED)
#     list(APPEND ADD_REQUIREMENTS component2)
# endif()
# if(CONFIG_COMPONENT3_ENABLED)
#     list(APPEND ADD_REQUIREMENTS component3)
# endif()
#

###### Add link search path for requirements/libs ######
# list(APPEND ADD_LINK_SEARCH_PATH "${CONFIG_TOOLCHAIN_PATH}/lib")
# list(APPEND ADD_REQUIREMENTS pthread m)  # add system libs, pthread and math lib for example here
# set (OpenCV_DIR opencv/lib/cmake/opencv4)
# find_package(OpenCV REQUIRED)
###############################################

############ Add static libs ##################
# list(APPEND ADD_STATIC_LIB "lib/libtest.a")
###############################################

############ Add dynamic libs ##################
# list(APPEND ADD_DYNAMIC_LIB "lib/arch/v831/libmaix_nn.so"
#                             "lib/arch/v831/libmaix_cam.so"
# )
###############################################

#### Add compile option for this component ####
#### and components denpend on this component
# list(APPEND ADD_DEFINITIONS -DAAAAA222=1
#                             -DAAAAA333=1)
###############################################

############ Add static libs ##################
#### Update parent's variables like CMAKE_C_LINK_FLAGS
# set(CMAKE_C_LINK_FLAGS "${CMAKE_C_LINK_FLAGS} -Wl,--start-group libmaix/libtest.a -ltest2 -Wl,--end-group" PARENT_SCOPE)
###############################################

register_component(DYNAMIC)