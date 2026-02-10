cmake_minimum_required(VERSION 3.7.2)

# This project builds a Python extension only; do not generate the SDK's default executable target.
set(SKIP_PROJECT_EXECUTABLE ON)

# SDK path can be provided via environment variable.
set(SDK_ENV_NAME "MY_SDK_PATH")
set(SDK_PATH_ENV $ENV{${SDK_ENV_NAME}})

# Resolve SDK_PATH
if(NOT SDK_PATH)
    # Try common layouts near the project.
    get_filename_component(_sdk1 "${CMAKE_CURRENT_LIST_DIR}/../Auto-bind-Python-from-CC--main/Auto-bind-Python-from-CC--main" ABSOLUTE)
    get_filename_component(_sdk2 "${CMAKE_CURRENT_LIST_DIR}/../Auto-bind-Python-from-CC--main" ABSOLUTE)
    if(EXISTS "${_sdk1}/tools/cmake/project.py")
        set(SDK_PATH "${_sdk1}")
    elseif(EXISTS "${_sdk2}/tools/cmake/project.py")
        set(SDK_PATH "${_sdk2}")
    endif()

    if(SDK_PATH_ENV)
        if(EXISTS "${SDK_PATH_ENV}/tools/cmake/project.py")
            set(SDK_PATH "${SDK_PATH_ENV}")
        else()
            message(FATAL_ERROR "Env variable '${SDK_ENV_NAME}' is set, but '${SDK_PATH_ENV}' doesn't look like the SDK root")
        endif()
    endif()
endif()

if(NOT SDK_PATH OR NOT EXISTS "${SDK_PATH}/tools/cmake/compile.cmake")
    message(FATAL_ERROR "SDK path error. Please set MY_SDK_PATH to the SDK root (containing tools/cmake/compile.cmake)")
endif()

# -----------------------------------------------------------------------------
# Safety guard: never treat the component folder `main/` as a standalone project.
# If you accidentally run the SDK build tool inside `rvv_ext/main`, it would
# generate configs under `rvv_ext/main/build/...` repeatedly and look like a loop.
# -----------------------------------------------------------------------------
get_filename_component(_this_dir_name "${CMAKE_CURRENT_LIST_DIR}" NAME)
if(_this_dir_name STREQUAL "main")
    message(FATAL_ERROR "Please run build from the project root (rvv_ext), not from rvv_ext/main")
endif()

# Force project root path (used by SDK Kconfig via env PROJECT_PATH).
set(PROJECT_PATH "${CMAKE_CURRENT_LIST_DIR}" CACHE PATH "Project root" FORCE)

# Include SDK build framework.
include(${SDK_PATH}/tools/cmake/compile.cmake)

# SDK tools parse this line to get project name.
get_filename_component(parent_dir ${CMAKE_PARENT_LIST_FILE} DIRECTORY)
get_filename_component(project_dir_name ${parent_dir} NAME)
set(PROJECT_NAME ${project_dir_name})
message("-- PROJECT_NAME:${PROJECT_NAME}")
project(${PROJECT_NAME})

# -----------------------------------------------------------------------------
# IMPORTANT: ensure the 'main' component is part of the build graph.
#
# The SDK normally discovers components and adds them with:
#   add_subdirectory(<src>/main main EXCLUDE_FROM_ALL)
# which uses the binary dir name 'main' => ${PROJECT_BINARY_DIR}/main.
#
# If we add it again with the same binary dir, CMake will error out.
# Therefore, only add it manually when the SDK did NOT create the `main` target.
# -----------------------------------------------------------------------------
if(NOT TARGET main AND EXISTS "${CMAKE_CURRENT_LIST_DIR}/main/CMakeLists.txt")
    add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/main" "project_main" EXCLUDE_FROM_ALL)
endif()