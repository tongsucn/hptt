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

macro(hptc_set_lib_install LIB_NAME)
  # Shared/static library
  set(HPTC_LIB_INSTALL_TARGET_DIR lib)
  install(TARGETS ${LIB_NAME}
    LIBRARY DESTINATION ${HPTC_LIB_INSTALL_TARGET_DIR})

  message("-- Shared library lib" ${LIB_NAME} ".so will be installed in "
    ${CMAKE_INSTALL_PREFIX}/${HPTC_LIB_INSTALL_TARGET_DIR})
endmacro()

macro(hptc_set_lib_static_install LIB_NAME)
  # Shared/static library
  set(HPTC_LIB_INSTALL_TARGET_DIR lib)
  install(TARGETS ${LIB_NAME}
    ARCHIVE DESTINATION ${HPTC_LIB_INSTALL_TARGET_DIR})

  message("-- Static library lib" ${LIB_NAME} ".so will be installed in "
    ${CMAKE_INSTALL_PREFIX}/${HPTC_LIB_INSTALL_TARGET_DIR})
endmacro()
