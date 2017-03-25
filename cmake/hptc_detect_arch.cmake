# ----------------------------------------------------------------------------
# Architecture detection configuration
# ----------------------------------------------------------------------------
string(CONCAT HPTC_ARCH_DETECT_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}
  "/cmake/pylib/cpu_flags.py")

# ----------------------------------------------------------------------------
# Architecture detection
# ----------------------------------------------------------------------------
macro(hptc_detect_arch)
  execute_process(
    COMMAND ${HPTC_PYTHON_EXEC} ${HPTC_ARCH_DETECT_SCRIPT}
    OUTPUT_VARIABLE DETECT_STDOUT
    ERROR_VARIABLE DETECT_STDERR
    RESULT_VARIABLE RET_VAL
    )

  if (NOT RET_VAL EQUAL 0)
    message("!! Failed to detect architecture.")
    message("!! Detection script stdout:\n" ${DETECT_STDOUT})
    message("!! Detection script stderr:\n" ${DETECT_STDERR})
    message(WARNING "!! HPTC will enable trivial implementation.")
  else ()
    message("-- Selected architecture: " ${DETECT_STDOUT})
    set(HPTC_ARCH_TYPE ${DETECT_STDOUT})
  endif ()
endmacro()
