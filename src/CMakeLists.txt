# ----------------------------------------------------------------------------
# Target dependencies setting
# ----------------------------------------------------------------------------
# General files
file(GLOB HPTC_IMPL_SRC impl/*.cc)
file(GLOB HPTC_KERNEL_SRC kernels/*.cc)
file(GLOB HPTC_UTIL_SRC util/*.cc)
file(GLOB HPTC_GENERATED_SRC hptc/gen/*.cc)
file(GLOB HPTC_GENERAL_SRC
  ${HPTC_IMPL_SRC}
  ${HPTC_KERNEL_SRC}
  ${HPTC_UTIL_SRC}
  ${HPTC_GENERATED_SRC})

# Loader
file(GLOB HPTC_SRC arch/*.cc hptc.cc)

# Architecture related
file(GLOB HPTC_ARCH_AVX2_SRC arch/avx2/*.cc)
file(GLOB HPTC_ARCH_AVX_SRC arch/avx/*.cc)
file(GLOB HPTC_ARCH_IBM_SRC arch/ibnm/*.cc)
file(GLOB HPTC_ARCH_COMMON_SRC arch/common/*.cc)

file(GLOB HPTC_AVX2_SRC ${HPTC_GENERAL_SRC} ${HPTC_ARCH_AVX2_SRC})
file(GLOB HPTC_AVX_SRC ${HPTC_GENERAL_SRC} ${HPTC_ARCH_AVX_SRC})
file(GLOB HPTC_IBM_SRC ${HPTC_GENERAL_SRC} ${HPTC_ARCH_IBM_SRC})
file(GLOB HPTC_COMMON_SRC ${HPTC_GENERAL_SRC} ${HPTC_ARCH_COMMON_SRC})

# Benchmark dependencies
file(GLOB HPTC_BENCHMARK_SRC util/*.cc)


# ----------------------------------------------------------------------------
# Set shared and static libraries
# ----------------------------------------------------------------------------
# Loader shared/static libraries
add_library(${HPTC_LIB_NAME} SHARED ${HPTC_SRC})
target_link_libraries(${HPTC_LIB_NAME} dl)

add_library(${HPTC_LIB_STATIC_NAME} STATIC ${HPTC_SRC})
target_link_libraries(${HPTC_LIB_STATIC_NAME} dl)
set_target_properties(${HPTC_LIB_STATIC_NAME} PROPERTIES OUTPUT_NAME
  ${HPTC_LIB_NAME})


# AVX2 architecture
if (NOT ${HPTC_ARCH_AVX2} STREQUAL "")
  add_library(${HPTC_AVX2_LIB_NAME} SHARED ${HPTC_AVX2_SRC})

  # Set architecture-related compiler options
  if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    target_compile_options(${HPTC_AVX2_LIB_NAME}
      PUBLIC -march=core-avx2 ${HPTC_ARCH_AVX2})
  elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    target_compile_options(${HPTC_AVX2_LIB_NAME}
      PUBLIC -mavx2 ${HPTC_ARCH_AVX2})
  endif ()

  get_target_property(HPTC_AVX2_FLAGS ${HPTC_AVX2_LIB_NAME} COMPILE_OPTIONS)
  message("-- Additional compiler flags for AVX2 library: " ${HPTC_AVX2_FLAGS})

  # Add dependencies
  add_dependencies(${HPTC_LIB_NAME} ${HPTC_AVX2_LIB_NAME})
  add_dependencies(${HPTC_LIB_STATIC_NAME} ${HPTC_AVX2_LIB_NAME})

  # Set install
  install(TARGETS ${HPTC_AVX2_LIB_NAME} LIBRARY DESTINATION lib)
endif ()


# AVX architecture
if (NOT ${HPTC_ARCH_AVX} STREQUAL "")
  add_library(${HPTC_AVX_LIB_NAME} SHARED ${HPTC_AVX_SRC})

  # Set architecture-related compiler options
  target_compile_options(${HPTC_AVX_LIB_NAME} PUBLIC -mavx ${HPTC_ARCH_AVX})

  get_target_property(HPTC_AVX_FLAGS ${HPTC_AVX_LIB_NAME} COMPILE_OPTIONS)
  message("-- Additional compiler flags for AVX library: " ${HPTC_AVX_FLAGS})

  # Add dependencies and linkage
  add_dependencies(${HPTC_LIB_NAME} ${HPTC_AVX_LIB_NAME})
  add_dependencies(${HPTC_LIB_STATIC_NAME} ${HPTC_AVX_LIB_NAME})

  # Set install
  install(TARGETS ${HPTC_AVX_LIB_NAME} LIBRARY DESTINATION lib)
endif ()


# IBM architecture
if (NOT ${HPTC_ARCH_IBM} STREQUAL "")
  add_library(${HPTC_IBM_LIB_NAME} SHARED ${HPTC_IBM_SRC})

  # Add dependencies
  add_dependencies(${HPTC_LIB_NAME} ${HPTC_IBM_LIB_NAME})
  add_dependencies(${HPTC_LIB_STATIC_NAME} ${HPTC_IBM_LIB_NAME})

  # Set install
  install(TARGETS ${HPTC_IBM_LIB_NAME} LIBRARY DESTINATION lib)
endif ()


# Common architecture
add_library(${HPTC_COMMON_LIB_NAME} SHARED ${HPTC_COMMON_SRC})
add_dependencies(${HPTC_LIB_NAME} ${HPTC_COMMON_LIB_NAME})
add_dependencies(${HPTC_LIB_STATIC_NAME} ${HPTC_COMMON_LIB_NAME})


# Set install
install(TARGETS
  ${HPTC_LIB_NAME}
  ${HPTC_LIB_STATIC_NAME}
  ${HPTC_COMMON_LIB_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
message("-- Libraries will be installed in " ${CMAKE_INSTALL_PREFIX}/lib)


# Benchmark dependencies
if (${HPTC_BUILD_BENCHMARK})
  add_library(${HPTC_BENCHMARK_LIB_NAME} STATIC ${HPTC_BENCHMARK_SRC})
  add_dependencies(${HPTC_BENCHMARK_LIB_NAME} ${HPTC_LIB_STATIC_NAME})
  message("-- Benchmark's library will be built...")
endif ()
