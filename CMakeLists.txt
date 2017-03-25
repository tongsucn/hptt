# ----------------------------------------------------------------------------
#  Root CMake script for Tensor Transpose Library
#
#    From the off-tree build directory, invoke:
#      $ cmake <PATH_TO_THIS_LIBRARY>
#
# ----------------------------------------------------------------------------
cmake_minimum_required (VERSION 3.4)
project (HPTC)

# Include macros and functions
set(HPTC_CMAKE_SCRIPT_DIR "cmake")

# Find dependencies
find_package(Doxygen)
find_package(GTest)

# Setting options
include(${HPTC_CMAKE_SCRIPT_DIR}/hptc_config.cmake)
hptc_set_options()
include(${HPTC_CMAKE_SCRIPT_DIR}/hptc_detect_arch.cmake)
hptc_detect_arch()
hptc_set_compiler()

# Call code generation script
include(${HPTC_CMAKE_SCRIPT_DIR}/hptc_code_gen.cmake)
hptc_code_gen_all(${CMAKE_CURRENT_SOURCE_DIR})

# Directory setting
include(cmake/hptc_dir_setting.cmake)
hptc_set_dir()

# Installation setting
include(${HPTC_CMAKE_SCRIPT_DIR}/hptc_install.cmake)
hptc_set_header_install()
