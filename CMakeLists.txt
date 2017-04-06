# ----------------------------------------------------------------------------
#  Root CMake script for Tensor Transpose Library
#
#    From the off-tree build directory, invoke:
#      $ cmake <PATH_TO_THIS_LIBRARY>
#
# ----------------------------------------------------------------------------
cmake_minimum_required (VERSION 3.4)
project (HPTT)

# Include macros and functions
set(HPTT_CMAKE_SCRIPT_DIR "cmake")

# Find dependencies
# find_package(Doxygen)
find_package(GTest)

# Setting options
include(${HPTT_CMAKE_SCRIPT_DIR}/hptt_config.cmake)
hptt_set_options()
include(${HPTT_CMAKE_SCRIPT_DIR}/hptt_detect_arch.cmake)
hptt_detect_arch()
hptt_set_compiler()

# Call code generation script
include(${HPTT_CMAKE_SCRIPT_DIR}/hptt_code_gen.cmake)
hptt_code_gen_all(${CMAKE_CURRENT_SOURCE_DIR})

# Initialize build targets setting
include(${HPTT_CMAKE_SCRIPT_DIR}/hptt_lib_setting.cmake)
hptt_init_lib_setting()

# Directory setting
include(cmake/hptt_dir_setting.cmake)
hptt_set_dir()

# Installation setting
include(${HPTT_CMAKE_SCRIPT_DIR}/hptt_install.cmake)
hptt_set_header_install()
