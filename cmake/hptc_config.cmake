# ----------------------------------------------------------------------------
# Fix variable definition
# ----------------------------------------------------------------------------
set(HPTC_PYTHON_EXEC "python3")
set(HPTC_CMAKE_INFO_PREFIX "-- ")

# ----------------------------------------------------------------------------
# Information strings
# ----------------------------------------------------------------------------
set(HPTC_MSG_CODE_GEN_TRANS_ORDER_MIN
  "Minimum order pre-compiled in transpose libaray, valid minimum is 2")

set(HPTC_MSG_CODE_GEN_TRANS_ORDER_MAX
  "Maximum order pre-compiled in transpose libaray, must be larger than \
minimum order")

set(HPTC_MSG_CODE_GEN_TRANS_DTYPE
  "Float type pre-compiled in transpose library, support: s (float), \
d (double), c (FloatComplex), z (DoubleComplex)")


# ----------------------------------------------------------------------------
# Architecture detection variables
# ----------------------------------------------------------------------------
set(HPTC_ARCH_AVX2 "")
set(HPTC_ARCH_AVX "")


# ----------------------------------------------------------------------------
# Code generation variables
# ----------------------------------------------------------------------------
set(HPTC_CODE_GEN_TARGET_DIR_ARG "--target")
set(HPTC_CODE_GEN_TRANS_ORDER_MIN_ARG "--order-min")
set(HPTC_CODE_GEN_TRANS_ORDER_MAX_ARG "--order-max")
set(HPTC_CODE_GEN_TRANS_DTYPE_ARG "--dtype")

set(HPTC_CODE_GEN_TARGET_DIR "src/hptc/gen")


# ----------------------------------------------------------------------------
# HPTC cmake options
# ----------------------------------------------------------------------------
macro(hptc_set_options)
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
  set(HPTC_CODE_GEN_TRANS_ORDER_MAX 6 CACHE STRING
    ${HPTC_MSG_CODE_GEN_TRANS_ORDER_MAX})

  # Data type of transpose pre-compiled in shared/static library.
  set(HPTC_CODE_GEN_TRANS_DTYPE "s,d,c,z" CACHE STRING
    ${HPTC_MSG_CODE_GEN_TRANS_DTYPE})

  string(CONCAT HPTC_CODE_GEN_TARGET_DIR ${CMAKE_CURRENT_SOURCE_DIR} "/"
    ${HPTC_CODE_GEN_TARGET_DIR})
endmacro()


# ----------------------------------------------------------------------------
# Compiler options
# ----------------------------------------------------------------------------
macro(hptc_set_compiler)
  # For now, we support only Intel compiler (version 16+)
  set(HPTC_CXX_FLAG "-Wall -Werror -std=c++14")
  set(HPTC_DEBUG_FLAG "-O0 -g")
  set(HPTC_RELEASE_FLAG "-O3 -DNDEBUG")

  if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    string(CONCAT HPTC_CXX_FLAG ${HPTC_CXX_FLAG} " -qopenmp -xhost")
    string(CONCAT HPTC_DEBUG_FLAG ${HPTC_DEBUG_FLAG} " -xhost -debug full")
  elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    string(CONCAT HPTC_CXX_FLAG ${HPTC_CXX_FLAG} " -fopenmp -march=native")
    string(CONCAT HPTC_DEBUG_FLAG ${HPTC_DEBUG_FLAG} " -ggdb -march=native")
  else ()
    message(FATAL_ERROR "!! Unrecognized compiler: " ${CMAKE_CXX_COMPILER_ID})
  endif ()

  if (${HPTC_BUILD_DEBUG})
    # Use debug configurations.
    string(CONCAT HPTC_CXX_FLAG ${HPTC_CXX_FLAG} " " ${HPTC_DEBUG_FLAG})
  else ()
    # Use release configuration.
    string(CONCAT HPTC_CXX_FLAG ${HPTC_CXX_FLAG} " " ${HPTC_RELEASE_FLAG})
  endif ()
  string(CONCAT CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} " " ${HPTC_CXX_FLAG})

  message("-- Compiler flags:" ${CMAKE_CXX_FLAGS})
endmacro()
