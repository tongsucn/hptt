# ----------------------------------------------------------------------------
# Add transpose AVX benchmark files
# ----------------------------------------------------------------------------
set(HPTT_SELECTED_ARCH avx)
set(BM_TRANS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${HPTT_SELECTED_ARCH})
set(BM_TRANS_SRC_SUFFIX .cc)

file(GLOB BENCHMARK_SRC RELATIVE ${BM_TRANS_SRC}
    ${BM_TRANS_SRC}/*${BM_TRANS_SRC_SUFFIX})

foreach(BENCHMARK_CC_FILE ${BENCHMARK_SRC})
  string(REPLACE ${BM_TRANS_SRC_SUFFIX} "" BENCHMARK_NAME ${BENCHMARK_CC_FILE})
  add_executable(${BENCHMARK_NAME} ${BM_TRANS_SRC}/${BENCHMARK_CC_FILE})
  target_link_libraries(${BENCHMARK_NAME} ${HPTT_LIB_STATIC_NAME}
    ${HPTT_BENCHMARK_LIB_NAME})

  # Set special compiler flags
  if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    target_compile_options(${BENCHMARK_NAME} PUBLIC -xhost)
  elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    target_compile_options(${BENCHMARK_NAME} PUBLIC -march=native)
  endif ()
endforeach(BENCHMARK_CC_FILE ${BENCHMARK_SRC})
