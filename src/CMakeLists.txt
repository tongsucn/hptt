# ----------------------------------------------------------------------------
# Target dependencies setting
# ----------------------------------------------------------------------------
# General files
file(GLOB HPTT_IMPL_SRC impl/*.cc)
file(GLOB HPTT_KERNEL_SRC kernels/*.cc)
file(GLOB HPTT_UTIL_SRC util/*.cc)
file(GLOB HPTT_GENERATED_SRC hptt/gen/*.cc)
file(GLOB HPTT_GENERAL_SRC
  ${HPTT_IMPL_SRC}
  ${HPTT_KERNEL_SRC}
  ${HPTT_UTIL_SRC}
  ${HPTT_GENERATED_SRC})

# Loader
file(GLOB HPTT_SRC arch/*.cc hptt.cc)

# Architecture related
file(GLOB HPTT_ARCH_AVX2_SRC arch/avx2/*.cc)
file(GLOB HPTT_ARCH_AVX_SRC arch/avx/*.cc)
file(GLOB HPTT_ARCH_ARM_SRC arch/arm/*.cc)
file(GLOB HPTT_ARCH_IBM_SRC arch/ibm/*.cc)
file(GLOB HPTT_ARCH_COMMON_SRC arch/common/*.cc)

file(GLOB HPTT_AVX2_SRC ${HPTT_GENERAL_SRC} ${HPTT_ARCH_AVX2_SRC})
file(GLOB HPTT_AVX_SRC ${HPTT_GENERAL_SRC} ${HPTT_ARCH_AVX_SRC})
file(GLOB HPTT_ARM_SRC ${HPTT_GENERAL_SRC} ${HPTT_ARCH_ARM_SRC})
file(GLOB HPTT_IBM_SRC ${HPTT_GENERAL_SRC} ${HPTT_ARCH_IBM_SRC})
file(GLOB HPTT_COMMON_SRC ${HPTT_GENERAL_SRC} ${HPTT_ARCH_COMMON_SRC})

# Benchmark dependencies
file(GLOB HPTT_BENCHMARK_SRC util/*.cc)


# ----------------------------------------------------------------------------
# Set shared and static libraries
# ----------------------------------------------------------------------------
# Loader shared/static libraries
add_library(${HPTT_LIB_NAME} SHARED ${HPTT_SRC})
target_link_libraries(${HPTT_LIB_NAME} dl)

add_library(${HPTT_LIB_STATIC_NAME} STATIC ${HPTT_SRC})
target_link_libraries(${HPTT_LIB_STATIC_NAME} dl)
set_target_properties(${HPTT_LIB_STATIC_NAME} PROPERTIES OUTPUT_NAME
  ${HPTT_LIB_NAME})


# AVX2 architecture
if (NOT ${HPTT_ARCH_AVX2} STREQUAL "")
  add_library(${HPTT_AVX2_LIB_NAME} SHARED ${HPTT_AVX2_SRC})

  # Set architecture-related compiler options
  if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    target_compile_options(${HPTT_AVX2_LIB_NAME}
      PUBLIC -march=core-avx2 ${HPTT_ARCH_AVX2})
  elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    target_compile_options(${HPTT_AVX2_LIB_NAME}
      PUBLIC -mavx2 ${HPTT_ARCH_AVX2})
  endif ()

  get_target_property(HPTT_AVX2_FLAGS ${HPTT_AVX2_LIB_NAME} COMPILE_OPTIONS)
  message("-- Additional compiler flags for AVX2 library: " ${HPTT_AVX2_FLAGS})

  # Add dependencies
  add_dependencies(${HPTT_LIB_NAME} ${HPTT_AVX2_LIB_NAME})
  add_dependencies(${HPTT_LIB_STATIC_NAME} ${HPTT_AVX2_LIB_NAME})

  # Set install
  install(TARGETS ${HPTT_AVX2_LIB_NAME} LIBRARY DESTINATION lib)
endif ()


# AVX architecture
if (NOT ${HPTT_ARCH_AVX} STREQUAL "")
  add_library(${HPTT_AVX_LIB_NAME} SHARED ${HPTT_AVX_SRC})

  # Set architecture-related compiler options
  target_compile_options(${HPTT_AVX_LIB_NAME} PUBLIC -mavx ${HPTT_ARCH_AVX})

  get_target_property(HPTT_AVX_FLAGS ${HPTT_AVX_LIB_NAME} COMPILE_OPTIONS)
  message("-- Additional compiler flags for AVX library: " ${HPTT_AVX_FLAGS})

  # Add dependencies and linkage
  add_dependencies(${HPTT_LIB_NAME} ${HPTT_AVX_LIB_NAME})
  add_dependencies(${HPTT_LIB_STATIC_NAME} ${HPTT_AVX_LIB_NAME})

  # Set install
  install(TARGETS ${HPTT_AVX_LIB_NAME} LIBRARY DESTINATION lib)
endif ()


# ARM architecture
if (NOT ${HPTT_ARCH_ARM} STREQUAL "")
  add_library(${HPTT_ARM_LIB_NAME} SHARED ${HPTT_ARM_SRC})

  # Set architecture-related compiler options
  target_compile_options(${HPTT_ARM_LIB_NAME}
    PUBLIC -march=native ${HPTT_ARCH_ARM})

  get_target_property(HPTT_ARM_FLAGS ${HPTT_ARM_LIB_NAME} COMPILE_OPTIONS)
  message("-- Additional compiler flags for ARM library: " ${HPTT_ARM_FLAGS})

  # Add dependencies
  add_dependencies(${HPTT_LIB_NAME} ${HPTT_ARM_LIB_NAME})
  add_dependencies(${HPTT_LIB_STATIC_NAME} ${HPTT_ARM_LIB_NAME})

  # Set install
  install(TARGETS ${HPTT_ARM_LIB_NAME} LIBRARY DESTINATION lib)
endif ()


# IBM architecture
if (NOT ${HPTT_ARCH_IBM} STREQUAL "")
  add_library(${HPTT_IBM_LIB_NAME} SHARED ${HPTT_IBM_SRC})

  # Add dependencies
  add_dependencies(${HPTT_LIB_NAME} ${HPTT_IBM_LIB_NAME})
  add_dependencies(${HPTT_LIB_STATIC_NAME} ${HPTT_IBM_LIB_NAME})

  # Set install
  install(TARGETS ${HPTT_IBM_LIB_NAME} LIBRARY DESTINATION lib)
endif ()


# Common architecture
add_library(${HPTT_COMMON_LIB_NAME} SHARED ${HPTT_COMMON_SRC})
add_dependencies(${HPTT_LIB_NAME} ${HPTT_COMMON_LIB_NAME})
add_dependencies(${HPTT_LIB_STATIC_NAME} ${HPTT_COMMON_LIB_NAME})


# Set install
install(TARGETS
  ${HPTT_LIB_NAME}
  ${HPTT_LIB_STATIC_NAME}
  ${HPTT_COMMON_LIB_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
message("-- Libraries will be installed in " ${CMAKE_INSTALL_PREFIX}/lib)


# Benchmark dependencies
if (${HPTT_BUILD_BENCHMARK})
  add_library(${HPTT_BENCHMARK_LIB_NAME} STATIC ${HPTT_BENCHMARK_SRC})
  add_dependencies(${HPTT_BENCHMARK_LIB_NAME} ${HPTT_LIB_STATIC_NAME})
  message("-- Benchmark's library will be built...")
endif ()
