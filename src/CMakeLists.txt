# ----------------------------------------------------------------------------
# Dependencies setting
# ----------------------------------------------------------------------------
file(GLOB HPTC_ROOT_SRC *.cc)
file(GLOB HPTC_KERNEL_SRC kernels/*.cc)
file(GLOB HPTC_GENERATED gen/*.cc)

# ----------------------------------------------------------------------------
# Libraries setting
# ----------------------------------------------------------------------------
# Shared libraries
add_library(
  hptc SHARED
  ${HPTC_ROOT_SRC}
  ${HPTC_KERNEL_SRC}
  ${HPTC_GENERATED})

add_library(
  hptc_static STATIC
  ${HPTC_ROOT_SRC}
  ${HPTC_KERNEL_SRC}
  ${HPTC_GENERATED})

set_target_properties(hptc_static PROPERTIES OUTPUT_NAME hptc)


# ----------------------------------------------------------------------------
# Installation setting
# ----------------------------------------------------------------------------
# Configure installation
#install(TARGETS hptc hptc_static
#      LIBRARY DESTINATION lib
#      ARCHIVE DESTINATION lib)

