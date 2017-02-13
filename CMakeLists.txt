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
# Options
option(BUILD_TEST "Generate executable test program.")
option(BUILD_DOC "Create the HTML based API documentation (requires doxygen)."
  ${DOXYGEN_FOUND})
option(BUILD_TEST "Generate executable test program (requires Google Test)."
  ${GTEST_FOUND})
set(BUILD_DEBUG OFF CACHE BOOL "Build debug version of library.")

# Find dependencies
find_package(Doxygen)
find_package(GTest)


# ----------------------------------------------------------------------------
# Compiler options
# ----------------------------------------------------------------------------
set(TENSOR_CXX_FLAG "-Wall -Werror -std=c++14 -xhost -qopenmp")
set(TENSOR_DEBUG_FLAG "-O0 -g")
set(TENSOR_RELEASE_FLAG "-O3 -DNDEBUG")

if (BUILD_TYPE STREQUAL "DEBUG")
  # Use debug configurations.
  string(CONCAT TENSOR_CXX_FLAG ${TENSOR_CXX_FLAG} " " ${TENSOR_DEBUG_FLAG})
else ()
  # Use release configuration.
  string(CONCAT TENSOR_CXX_FLAG ${TENSOR_CXX_FLAG} " " ${TENSOR_RELEASE_FLAG})
endif ()

string(CONCAT CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} " " ${TENSOR_CXX_FLAG})


# ----------------------------------------------------------------------------
# Directory setting
# ----------------------------------------------------------------------------
# Add include files' directory
include_directories(inc)

# Add source directory
add_subdirectory(src)

# ----------------------------------------------------------------------------
# Tests and benchmarks
# ----------------------------------------------------------------------------
if (BUILD_TEST AND GTEST_FOUND)
  add_subdirectory(test)
endif ()

if (BUILD_BENCHMARK)
  if (NOT BUILD_TEST)
    add_subdirectory(test)
  endif ()

  add_subdirectory(benchmark)
endif ()

#if (BUILD_DOC)
#  add_subdirectory(doc)
#endif()

