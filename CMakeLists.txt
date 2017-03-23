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
include(cmake/hptc_config.cmake)
include(cmake/hptc_code_gen.cmake)

# Find dependencies
find_package(Doxygen)
find_package(GTest)

# ----------------------------------------------------------------------------
# Build options
# ----------------------------------------------------------------------------
option(HPTC_BUILD_DOC
  "Create the HTML based API documentation (requires doxygen)."
  ${DOXYGEN_FOUND})
option(HPTC_BUILD_DEBUG
  "Generate executable test program (requires Google Test)." OFF)
option(HPTC_BUILD_BENCHMARK
  "Generate benchmark executable." OFF)
option(HPTC_BUILD_TEST
  "Generate executable test program (requires Google Test)." OFF)

# Minimum order number of transpose pre-compiled in shared/static library.
set(HPTC_CODE_GEN_TRANS_ORDER_MIN 2 CACHE STRING
  ${HPTC_MSG_CODE_GEN_TRANS_ORDER_MIN})

# Maximum order number of transpose pre-compiled in shared/static library.
set(HPTC_CODE_GEN_TRANS_ORDER_MAX 7 CACHE STRING
  ${HPTC_MSG_CODE_GEN_TRANS_ORDER_MAX})

# Data type of transpose pre-compiled in shared/static library.
set(HPTC_CODE_GEN_TRANS_DTYPE "s,d,c,z" CACHE STRING
  ${HPTC_MSG_CODE_GEN_TRANS_DTYPE})

# Coefficient usage type of transpose pre-compiled in shared/static library.
set(HPTC_CODE_GEN_TRANS_COEF "both" CACHE STRING
  ${HPTC_MSG_CODE_GEN_TRANS_COEF})

set(HPTC_CODE_GEN_TARGET_DIR_PREFIX ${CMAKE_CURRENT_SOURCE_DIR} CACHE PATH
  ${HPTC_MSG_CODE_GEN_TARGET_DIR_PREFIX})

string(CONCAT HPTC_CODE_GEN_TARGET_DIR ${HPTC_CODE_GEN_TARGET_DIR_PREFIX} "/"
  ${HPTC_CODE_GEN_TARGET_DIR})


# Setting compiler options
hptc_set_compiler_option()

# Call code generation script
hptc_code_gen_all(${HPTC_CODE_GEN_TARGET_DIR_PREFIX})

# ----------------------------------------------------------------------------
# Directory setting
# ----------------------------------------------------------------------------
# Add include files' directory
include_directories(inc)

# Add generated include files' directory
include_directories(${HPTC_CODE_GEN_TARGET_DIR})

# Add source directory
add_subdirectory(src)

# Add test and benchmark directory if necessary
add_subdirectory(test)

# Add documentation
#if (BUILD_DOC)
#  add_subdirectory(doc)
#endif()
