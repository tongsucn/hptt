# ----------------------------------------------------------------------------
# Dependencies setting
# ----------------------------------------------------------------------------
file(GLOB HPTC_ROOT_SRC tensor.cc)
file(GLOB HPTC_CGRAPH_SRC cgraph/*.cc)
file(GLOB HPTC_OPER_SRC operations/*.cc)
file(GLOB HPTC_PARAM_SRC param/*.cc)
file(GLOB HPTC_KERNEL_SRC kernels/*.cc)

# ----------------------------------------------------------------------------
# Libraries setting
# ----------------------------------------------------------------------------
# Shared libraries
add_library(hptc SHARED ${HPTC_ROOT_SRC} ${HPTC_CGRAPH_SRC} ${HPTC_OPER_SRC}
    ${HPTC_PARAM_SRC} ${HPTC_KERNEL_SRC})

# ----------------------------------------------------------------------------
# Installation setting
# ----------------------------------------------------------------------------
# Configure installation
#install(TARGETS hptc hptc_static
#      LIBRARY DESTINATION lib
#      ARCHIVE DESTINATION lib)

