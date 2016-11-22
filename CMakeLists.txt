# ----------------------------------------------------------------------------
#  Root CMake script for Tensor Transpose Library
#
#    From the off-tree build directory, invoke:
#      $ cmake <PATH_TO_THIS_LIBRARY>
#
# ----------------------------------------------------------------------------
cmake_minimum_required (VERSION 3.4)
project (HPTC)


# ----------------------------------------------------------------------------
# Build options
# ----------------------------------------------------------------------------
# Find dependencies
find_package(Doxygen)
find_package(GTest)

# Options
option(BUILD_TEST "Generate executable test program.")
option(BUILD_DOC "Create the HTML based API documentation (requires doxygen)."
  ${DOXYGEN_FOUND})
option(BUILD_TEST "Generate executable test program (requires Google Test)."
  ${GTEST_FOUND})
set(BUILD_DEBUG OFF CACHE BOOL "Build debug version of library.")


# ----------------------------------------------------------------------------
# Compiler options
# ----------------------------------------------------------------------------
set(TENSOR_CXX_FLAGS "-Wall -Werror -std=c++14")
set(TENSOR_DEBUG_FLAG "-g")
set(TENSOR_RELEASE_FLAG "-O3 -DNDEBUG")

if (BUILD_TYPE STREQUAL "DEBUG")
  # Use debug configurations.
  set(TENSOR_CXX_FLAGS "${TENSOR_CXX_FLAGS} ${TENSOR_DEBUG_FLAG}")
else ()
  # Use release configuration.
  set(TENSOR_CXX_FLAGS "${TENSOR_CXX_FLAGS} ${TENSOR_RELEASE_FLAG}")
endif ()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TENSOR_CXX_FLAGS}")

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
if (BUILD_TEST AND GTEST_FOUND)
  add_subdirectory(test)
endif ()

if (BUILD_DOC)
  add_subdirectory(doc)
endif()

