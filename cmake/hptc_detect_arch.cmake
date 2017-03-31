# ----------------------------------------------------------------------------
# Architecture detection configuration
# ----------------------------------------------------------------------------
string(CONCAT HPTC_ARCH_DETECT_SCRIPT ${CMAKE_SOURCE_DIR}
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
    message("-- Enabled architecture kernels: common " ${DETECT_STDOUT})

    if (NOT ${DETECT_STDOUT} STREQUAL "")
      string(REPLACE " " ";" ARCH_ABBREV_LIST ${DETECT_STDOUT})
    endif ()

    # Set architecture C++ macros
    foreach (ARCH ${ARCH_ABBREV_LIST})
      if (${ARCH} STREQUAL "avx2")
        set(HPTC_ARCH_AVX2 "-DHPTC_ARCH_AVX2")
      elseif (${ARCH} STREQUAL "avx")
        set(HPTC_ARCH_AVX "-DHPTC_ARCH_AVX")
      endif ()
    endforeach ()
  endif ()
endmacro()
