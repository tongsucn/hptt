# ----------------------------------------------------------------------------
# Fix variable definition
# ----------------------------------------------------------------------------
set(HPTC_PYTHON_EXEC "python3")

# ----------------------------------------------------------------------------
# Architecture detection variables
# ----------------------------------------------------------------------------
set(HPTC_ARCH_AVX2 "")
set(HPTC_ARCH_AVX "")


# ----------------------------------------------------------------------------
# HPTC cmake options
# ----------------------------------------------------------------------------
macro(hptc_set_options)
  option(HPTC_BUILD_DOC
    "Create the HTML based API documentation (requires doxygen)."
    ${DOXYGEN_FOUND})
  option(HPTC_BUILD_DEBUG "Generate debug version libray." OFF)
  option(HPTC_BUILD_BENCHMARK "Build benchmark executable." OFF)
  option(HPTC_BUILD_TEST
    "Generate executable test program (requires Google Test)." OFF)

  # Minimum order number of transpose pre-compiled in shared/static library.
  set(HPTC_CODE_GEN_TRANS_ORDER_MIN 2 CACHE STRING
    "Minimum order pre-compiled in transpose libaray, valid minimum is 2")

  # Maximum order number of transpose pre-compiled in shared/static library.
  set(HPTC_CODE_GEN_TRANS_ORDER_MAX 6 CACHE STRING
    "Maximum order pre-compiled in transpose libaray, must be larger than \
minimum order")
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
    string(CONCAT HPTC_CXX_FLAG ${HPTC_CXX_FLAG} " -qopenmp -ipo")
    string(CONCAT HPTC_DEBUG_FLAG ${HPTC_DEBUG_FLAG} " -debug full")
  elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    string(CONCAT HPTC_CXX_FLAG ${HPTC_CXX_FLAG}
      " -fopenmp -fsanitize=address")
    string(CONCAT HPTC_DEBUG_FLAG ${HPTC_DEBUG_FLAG}
      " -ggdb -fsanitize=address")
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
