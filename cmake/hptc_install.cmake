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
