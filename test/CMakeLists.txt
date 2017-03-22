# ----------------------------------------------------------------------------
# Directory setting
# ----------------------------------------------------------------------------
if (${HPTC_BUILD_BENCHMARK} OR ${HPTC_BUILD_TEST})
  include_directories(inc)
endif ()

# Add test and benchmark directory if necessary
if (${HPTC_BUILD_BENCHMARK})
  add_subdirectory(benchmark)
endif ()

if (${HPTC_BUILD_TEST})
  if (${GTEST_FOUND})
    add_subdirectory(test)
  else ()
    message(FATAL_ERROR "To build test or benchmark, Google Test is required.")
  endif ()
endif ()
