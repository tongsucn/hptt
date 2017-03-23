# ----------------------------------------------------------------------------
# Installation configuration
# ----------------------------------------------------------------------------
macro(hptc_set_header_install)
  # Header files
  install(DIRECTORY inc/hptc
    DESTINATION "include")

  # Generated header files
  file(GLOB HPTC_GEN_HEADERS ${HPTC_CODE_GEN_TARGET_DIR}/*.tcc)
  install(FILES ${HPTC_GEN_HEADERS}
    DESTINATION "include/hptc/gen")

  message("-- Header files will be installed in "
    ${CMAKE_INSTALL_PREFIX}/include/hptc)
endmacro()

macro(hptc_set_target_install)
  # Shared/static library
  set(HPTC_LIB_INSTALL_TARGET_DIR lib)
  install(TARGETS hptc hptc_static
    LIBRARY DESTINATION ${HPTC_LIB_INSTALL_TARGET_DIR}
    ARCHIVE DESTINATION ${HPTC_LIB_INSTALL_TARGET_DIR})

  message("-- Library will be installed in "
    ${CMAKE_INSTALL_PREFIX}/${HPTC_LIB_INSTALL_TARGET_DIR})
endmacro()
