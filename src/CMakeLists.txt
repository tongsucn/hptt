# Target dependencies setting
file(GLOB HPTC_KERNEL_SRC kernels/*.cc)
file(GLOB HPTC_UTIL_SRC util/*.cc)
file(GLOB HPTC_GENERATED gen/*.cc)

# Shared and static libraries
add_library(
  hptc SHARED
  ${HPTC_KERNEL_SRC}
  ${HPTC_UTIL_SRC}
  ${HPTC_GENERATED})

add_library(
  hptc_static STATIC
  ${HPTC_KERNEL_SRC}
  ${HPTC_UTIL_SRC}
  ${HPTC_GENERATED})

set_target_properties(hptc_static PROPERTIES OUTPUT_NAME hptc)

# Set install
include(../cmake/hptc_install.cmake)
hptc_set_target_install()
