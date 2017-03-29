# Library names
set(HPTC_LIB_NAME "hptc")
set(HPTC_LIB_STATIC_NAME "hptc_static")
set(HPTC_AVX2_LIB_NAME "hptc_avx2")
set(HPTC_AVX_LIB_NAME "hptc_avx")
set(HPTC_COMMON_LIB_NAME "hptc_common")


# Target dependencies setting
file(GLOB HPTC_SRC hptc*.cc)
file(GLOB HPTC_KERNEL_SRC kernels/*.cc)
file(GLOB HPTC_KERNEL_COMMON_SRC kernels/common/*.cc)
file(GLOB HPTC_KERNEL_AVX_SRC kernels/avx/*.cc)
file(GLOB HPTC_KERNEL_AVX2_SRC kernels/avx2/*.cc)
file(GLOB HPTC_UTIL_SRC util/*.cc)
file(GLOB HPTC_GENERATED hptc/gen/*.cc)


# Main shared library
add_library(${HPTC_LIB_NAME} SHARED
  ${HPTC_SRC}
  ${HPTC_KERNEL_SRC}
  ${HPTC_UTIL_SRC}
  ${HPTC_GENERATED})
target_link_libraries(${HPTC_LIB_NAME} dl)


# Main static library
add_library(${HPTC_LIB_STATIC_NAME} STATIC
  ${HPTC_SRC}
  ${HPTC_KERNEL_SRC}
  ${HPTC_UTIL_SRC}
  ${HPTC_GENERATED})
target_link_libraries(${HPTC_LIB_STATIC_NAME} dl)
set_target_properties(${HPTC_LIB_STATIC_NAME} PROPERTIES OUTPUT_NAME
  ${HPTC_LIB_NAME})
add_dependencies(${HPTC_LIB_NAME} ${HPTC_LIB_STATIC_NAME})


# AVX2 architecture
if (NOT ${HPTC_ARCH_AVX2} STREQUAL "")
  add_library(${HPTC_AVX2_LIB_NAME} SHARED ${HPTC_KERNEL_AVX2_SRC})
  add_dependencies(${HPTC_LIB_STATIC_NAME} ${HPTC_AVX2_LIB_NAME})
endif ()


# AVX architecture
if (NOT ${HPTC_ARCH_AVX} STREQUAL "")
  add_library(${HPTC_AVX_LIB_NAME} SHARED ${HPTC_KERNEL_AVX_SRC})
  add_dependencies(${HPTC_LIB_STATIC_NAME} ${HPTC_AVX_LIB_NAME})
endif ()


# Common architecture
add_library(${HPTC_COMMON_LIB_NAME} SHARED ${HPTC_KERNEL_COMMON_SRC})
add_dependencies(${HPTC_LIB_STATIC_NAME} ${HPTC_COMMON_LIB_NAME})


# Set install
install(TARGETS
  ${HPTC_LIB_NAME}
  ${HPTC_LIB_STATIC_NAME}
  ${HPTC_AVX2_LIB_NAME}
  ${HPTC_AVX_LIB_NAME}
  ${HPTC_COMMON_LIB_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
message("-- Libraries will be installed in " ${CMAKE_INSTALL_PREFIX}/lib)
