# ----------------------------------------------------------------------------
#  Root CMake script for Tensor Transpose Library
#
#    From the off-tree build directory, invoke:
#      $ cmake <PATH_TO_THIS_LIBRARY>
#
# ----------------------------------------------------------------------------
cmake_minimum_required (VERSION 3.4)
project (Tensor Computing Lib)


# ----------------------------------------------------------------------------
# Build options
# ----------------------------------------------------------------------------

# Find dependencies
find_package(Doxygen)

# Options
option(BUILD_TEST "Generate executable test program.")
option(BUILD_DOC "Create the HTML based API documentation (requires doxygen)."
  ${DOXYGEN_FOUND})
option(BUILD_TEST "Generate executable test program.")
set(BUILD_DEBUG OFF CACHE BOOL "Build debug version of library.")


# ----------------------------------------------------------------------------
# Directory setting
# ----------------------------------------------------------------------------
# Add include files' directory
include_directories(inc)

# Add source directory
add_subdirectory(src)


# ----------------------------------------------------------------------------
# Build setting
# ----------------------------------------------------------------------------
if (BUILD_TEST)
  add_subdirectory(test)
endif ()

if (BUILD_DOC)
  add_subdirectory(doc)
endif()


# ----------------------------------------------------------------------------
# Compiler options
# ----------------------------------------------------------------------------
set(TENSOR_CC_FLAGS "-Wall -std=c++14")
set(TENSOR_DEBUG_FLAG "")
set(TENSOR_RELEASE_FLAG "-O3 -DNDEBUG")

if (BUILD_TYPE STREQUAL "DEBUG")
  # Use debug configurations.
  set(TENSOR_CC_FLAGS "${TENSOR_CC_FALGS} ${TENSOR_DEBUG_FLAG}")
else ()
  # Use release configuration.
  set(TENSOR_CC_FLAGS "${TENSOR_CC_FALGS} ${TENSOR_RELEASE_FLAG}")
endif ()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${TENSOR_CC_FLAGS}")

