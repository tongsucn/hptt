# ----------------------------------------------------------------------------
# Fix variable definition
# ----------------------------------------------------------------------------
set(HPTT_PYTHON_EXEC "python3")

# ----------------------------------------------------------------------------
# Architecture detection variables
# ----------------------------------------------------------------------------
set(HPTT_ARCH_AVX2 "")
set(HPTT_ARCH_AVX "")


# ----------------------------------------------------------------------------
# HPTT cmake options
# ----------------------------------------------------------------------------
macro(hptt_set_options)
  option(HPTT_BUILD_DOC
    "Create the HTML based API documentation (requires doxygen)."
    ${DOXYGEN_FOUND})
  option(HPTT_BUILD_DEBUG "Generate debug version libray." OFF)
  option(HPTT_BUILD_BENCHMARK "Build benchmark executable." OFF)
  option(HPTT_BUILD_TEST
    "Generate executable test program (requires Google Test)." OFF)

  # Minimum order number of transpose pre-compiled in shared/static library.
  set(HPTT_CODE_GEN_TRANS_ORDER_MIN 2 CACHE STRING
    "Minimum order pre-compiled in transpose libaray, valid minimum is 2")

  # Maximum order number of transpose pre-compiled in shared/static library.
  set(HPTT_CODE_GEN_TRANS_ORDER_MAX 6 CACHE STRING
    "Maximum order pre-compiled in transpose libaray")
endmacro()


# ----------------------------------------------------------------------------
# Compiler options
# ----------------------------------------------------------------------------
macro(hptt_set_compiler)
  # For now, we support only Intel compiler (version 16+)
  set(HPTT_CXX_FLAG "-Wall -Werror -std=c++14")
  set(HPTT_DEBUG_FLAG "-O0 -g")
  set(HPTT_RELEASE_FLAG "-O3 -DNDEBUG")

  if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    set(HPTT_CXX_FLAG "${HPTT_CXX_FLAG} -qopenmp")
    set(HPTT_DEBUG_FLAG "${HPTT_DEBUG_FLAG} -debug full")
  elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    set(HPTT_CXX_FLAG "${HPTT_CXX_FLAG} -fopenmp")
  elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    set(HPTT_CXX_FLAG "${HPTT_CXX_FLAG} -fopenmp -fsanitize=address")
    set(HPTT_DEBUG_FLAG "${HPTT_DEBUG_FLAG} -ggdb -fsanitize=address")
  else ()
    message(FATAL_ERROR "!! Unrecognized compiler: " ${CMAKE_CXX_COMPILER_ID})
  endif ()

  if (${HPTT_BUILD_DEBUG})
    # Use debug configurations.
    set(HPTT_CXX_FLAG "${HPTT_CXX_FLAG} ${HPTT_DEBUG_FLAG}")
  else ()
    # Use release configuration.
    set(HPTT_CXX_FLAG "${HPTT_CXX_FLAG} ${HPTT_RELEASE_FLAG}")
  endif ()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  ${HPTT_CXX_FLAG}")

  message("-- Compiler flags:" ${CMAKE_CXX_FLAGS})
endmacro()
