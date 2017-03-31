# ----------------------------------------------------------------------------
# Code generation configuration
# ----------------------------------------------------------------------------
string(CONCAT HPTC_CODE_GEN_SCRIPT_DIR ${CMAKE_SOURCE_DIR}/cmake/pylib)
string(CONCAT HPTC_CODE_GEN_TRANS_SCRIPT ${HPTC_CODE_GEN_SCRIPT_DIR}
  "/code_gen_trans.py")

set(HPTC_CODE_GEN_TRANS_ORDER_MIN_ARG "--order-min")
set(HPTC_CODE_GEN_TRANS_ORDER_MAX_ARG "--order-max")
set(HPTC_CODE_GEN_TARGET_DIR_ARG "--target")
string(CONCAT HPTC_CODE_GEN_TARGET_DIR ${CMAKE_SOURCE_DIR} "/src/hptc/gen")


# ----------------------------------------------------------------------------
# Code generation
# ----------------------------------------------------------------------------
macro(hptc_code_gen_trans WORKING_DIR)
  message("-- Generating transpose library template explicit instantation. "
    "Target directory: " ${HPTC_CODE_GEN_TARGET_DIR})
  execute_process(
    COMMAND ${HPTC_PYTHON_EXEC} ${HPTC_CODE_GEN_TRANS_SCRIPT}
    ${HPTC_CODE_GEN_TARGET_DIR_ARG} ${HPTC_CODE_GEN_TARGET_DIR}
    ${HPTC_CODE_GEN_TRANS_ORDER_MIN_ARG} ${HPTC_CODE_GEN_TRANS_ORDER_MIN}
    ${HPTC_CODE_GEN_TRANS_ORDER_MAX_ARG} ${HPTC_CODE_GEN_TRANS_ORDER_MAX}
    WORKING_DIRECTORY ${WORKING_DIR}
    OUTPUT_VARIABLE CODE_GEN_STDOUT
    ERROR_VARIABLE CODE_GEN_STDERR
    RESULT_VARIABLE RET_VAL
    )

  if (NOT RET_VAL EQUAL 0)
    message("!! Failed to generate template explicit instantation.")
    message("!! Generation script stdout:\n" ${CODE_GEN_STDOUT})
    message("!! Generation script stderr:\n" ${CODE_GEN_STDERR})
    message(FATAL_ERROR "!! Library cannot be built without code generation.")
  endif ()
endmacro()

macro(hptc_code_gen_all WORKING_DIR)
  hptc_code_gen_trans(${WORKING_DIR})
endmacro()
