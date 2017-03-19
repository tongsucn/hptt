# ----------------------------------------------------------------------------
# Fix variable definition
# ----------------------------------------------------------------------------
set(HPTC_PYTHON_EXEC "python3")
set(HPTC_CMAKE_INFO_PREFIX "-- ")

# ----------------------------------------------------------------------------
# Information strings
# ----------------------------------------------------------------------------
string(CONCAT HPTC_MSG_CODE_GEN_TRANS ${HPTC_CMAKE_INFO_PREFIX}
  "Generating transpose library template explicit instantation.")

string(CONCAT HPTC_MSG_FATAL_CODE_GEN_TRANS ${HPTC_CMAKE_INFO_PREFIX}
  "Failed to generate template explicit instantation.")

set(HPTC_MSG_CODE_GEN_TRANS_ORDER_MIN
  "Minimum order pre-compiled in transpose libaray, valid minimum is 2")

set(HPTC_MSG_CODE_GEN_TRANS_ORDER_MAX
  "Maximum order pre-compiled in transpose libaray, must be larger than \
minimum order")

set(HPTC_MSG_CODE_GEN_TRANS_DTYPE
  "Float type pre-compiled in transpose library, support: s (float), \
d (double), c (FloatComplex), z (DoubleComplex)")

set(HPTC_MSG_CODE_GEN_TRANS_COEF
  "Coefficient usage pre-compiled in transpose library, support: alpha, beta, \
both, none")

set(HPTC_MSG_CODE_GEN_TARGET_DIR_PREFIX
  "Code generation target directory's prefix.")

# ----------------------------------------------------------------------------
# Code generation variables
# ----------------------------------------------------------------------------
set(HPTC_CODE_GEN_TARGET_DIR_ARG "--target")
set(HPTC_CODE_GEN_TRANS_ORDER_MIN_ARG "--order-min")
set(HPTC_CODE_GEN_TRANS_ORDER_MAX_ARG "--order-max")
set(HPTC_CODE_GEN_TRANS_DTYPE_ARG "--dtype")
set(HPTC_CODE_GEN_TRANS_COEF_ARG "--coef")

set(HPTC_CODE_GEN_TARGET_DIR "src/gen")


# ----------------------------------------------------------------------------
# Compiler options
# ----------------------------------------------------------------------------
# For now, we support only Intel compiler (version 16+)
set(HPTC_CXX_FLAG "-Wall -Werror -std=c++14 -xhost -qopenmp")
set(HPTC_DEBUG_FLAG "-O0 -g")
set(HPTC_RELEASE_FLAG "-O3 -DNDEBUG")

macro(hptc_set_compiler_option)
  if (${HPTC_BUILD_DEBUG})
    # Use debug configurations.
    string(CONCAT HPTC_CXX_FLAG ${HPTC_CXX_FLAG} " " ${HPTC_DEBUG_FLAG})
  else ()
    # Use release configuration.
    string(CONCAT HPTC_CXX_FLAG ${HPTC_CXX_FLAG} " " ${HPTC_RELEASE_FLAG})
  endif ()
  string(CONCAT CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} " " ${HPTC_CXX_FLAG})
endmacro()
