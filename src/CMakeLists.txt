# ----------------------------------------------------------------------------
# Dependencies setting
# ----------------------------------------------------------------------------
file(GLOB HPTC_ROOT_SRC *.cc)
file(GLOB HPTC_CGRAPH_SRC cgraph/*.cc)
file(GLOB HPTC_PLAN_SRC plan/*.cc)
file(GLOB HPTC_OPER_SRC operations/*.cc)
file(GLOB HPTC_PARAM_SRC param/*.cc)
file(GLOB HPTC_KERNEL_SRC kernels/*.cc)

# ----------------------------------------------------------------------------
# Libraries setting
# ----------------------------------------------------------------------------
# Shared libraries
add_library(hptc SHARED ${HPTC_ROOT_SRC} ${HPTC_CGRAPH_SRC} ${HPTC_PLAN_SRC}
    ${HPTC_OPER_SRC} ${HPTC_PARAM_SRC} ${HPTC_KERNEL_SRC})
add_library(hptc_static STATIC ${HPTC_ROOT_SRC} ${HPTC_CGRAPH_SRC}
    ${HPTC_PLAN_SRC} ${HPTC_OPER_SRC} ${HPTC_PARAM_SRC} ${HPTC_KERNEL_SRC})

set_target_properties(hptc_static PROPERTIES OUTPUT_NAME hptc)

# ----------------------------------------------------------------------------
# Installation setting
# ----------------------------------------------------------------------------
# Configure installation
#install(TARGETS hptc hptc_static
#      LIBRARY DESTINATION lib
#      ARCHIVE DESTINATION lib)

