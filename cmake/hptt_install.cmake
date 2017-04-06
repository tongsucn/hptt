# ----------------------------------------------------------------------------
# Installation configuration
# ----------------------------------------------------------------------------
macro(hptt_set_header_install)
  # Header files
  install(DIRECTORY inc/hptt DESTINATION "include")

  # Generated header files
  file(GLOB HPTT_GEN_HEADERS ${HPTT_CODE_GEN_TARGET_DIR}/*.tcc)
  install(FILES ${HPTT_GEN_HEADERS}
    DESTINATION "include/hptt/gen")

  message("-- Header files will be installed in "
    ${CMAKE_INSTALL_PREFIX}/include/hptt)
endmacro()
