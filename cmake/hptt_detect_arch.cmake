# ----------------------------------------------------------------------------
# Architecture detection configuration
# ----------------------------------------------------------------------------
set(HPTT_ARCH_DETECT_SCRIPT "${CMAKE_SOURCE_DIR}/cmake/pylib/cpu_flags.py")

# ----------------------------------------------------------------------------
# Architecture detection variables
# ----------------------------------------------------------------------------
set(HPTT_ARCH_AVX2 "")
set(HPTT_ARCH_AVX "")
set(HPTT_ARCH_ARM "")


# ----------------------------------------------------------------------------
# Architecture detection
# ----------------------------------------------------------------------------
macro(hptt_detect_arch)
  execute_process(
    COMMAND ${HPTT_PYTHON_EXEC} ${HPTT_ARCH_DETECT_SCRIPT}
    OUTPUT_VARIABLE DETECT_STDOUT
    ERROR_VARIABLE DETECT_STDERR
    RESULT_VARIABLE RET_VAL
    )

  if (NOT RET_VAL EQUAL 0)
    message("!! Failed to detect architecture.")
    message("!! Detection script stdout:\n" ${DETECT_STDOUT})
    message("!! Detection script stderr:\n" ${DETECT_STDERR})
    message(WARNING "!! HPTT will enable trivial implementation.")
  else ()
    message("-- Enabled architecture kernels: common " ${DETECT_STDOUT})

    if (NOT ${DETECT_STDOUT} STREQUAL "")
      string(REPLACE " " ";" ARCH_ABBREV_LIST ${DETECT_STDOUT})
    endif ()

    # Set architecture C++ macros
    foreach (ARCH ${ARCH_ABBREV_LIST})
      if (${ARCH} STREQUAL "avx2")
        set(HPTT_ARCH_AVX2 "-DHPTT_ARCH_AVX2")
      elseif (${ARCH} STREQUAL "avx")
        set(HPTT_ARCH_AVX "-DHPTT_ARCH_AVX")
      elseif (${ARCH} STREQUAL "arm")
        set(HPTT_ARCH_ARM "-DHPTT_ARCH_ARM")
      endif ()
    endforeach ()
    #set(HPTT_ARCH_IBM "-DHPTT_ARCH_IBM")
  endif ()
endmacro()
