# ----------------------------------------------------------------------------
# Directory setting
# ----------------------------------------------------------------------------
macro(hptc_set_dir)
  # Add header files and generated header files directory
  include_directories(inc src)

  # Add source directory
  add_subdirectory(src)

  # Add test and benchmark directory if requested by user
  set(HPTC_TEST_DIR "test")
  if (${HPTC_BUILD_BENCHMARK} OR ${HPTC_BUILD_TEST})
    include_directories(${HPTC_TEST_DIR}/inc)
  endif ()

  if (${HPTC_BUILD_BENCHMARK})
    message("-- Enable benchmarks building.")
    add_subdirectory(${HPTC_TEST_DIR}/benchmark)
  endif ()

  if (${HPTC_BUILD_TEST})
    if (${GTEST_FOUND})
      message("-- GoogleTest found, enable tests building.")
      add_subdirectory(${HPTC_TEST_DIR}/test)
    else ()
      message(FATAL_ERROR "!! Google Test is required for building tests.")
    endif ()
  endif ()

  # Add documentation
  #if (BUILD_DOC)
  #  add_subdirectory(doc)
  #endif()
endmacro()
