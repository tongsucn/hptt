# ----------------------------------------------------------------------------
# Code generation configuration
# ----------------------------------------------------------------------------
string(CONCAT HPTT_CODE_GEN_SCRIPT_DIR ${CMAKE_SOURCE_DIR}/cmake/pylib)
string(CONCAT HPTT_CODE_GEN_TRANS_SCRIPT ${HPTT_CODE_GEN_SCRIPT_DIR}
  "/code_gen_trans.py")

set(HPTT_CODE_GEN_TRANS_ORDER_MIN_ARG "--order-min")
set(HPTT_CODE_GEN_TRANS_ORDER_MAX_ARG "--order-max")
set(HPTT_CODE_GEN_TARGET_DIR_ARG "--target")
string(CONCAT HPTT_CODE_GEN_TARGET_DIR ${CMAKE_SOURCE_DIR} "/src/hptt/gen")


# ----------------------------------------------------------------------------
# Code generation
# ----------------------------------------------------------------------------
macro(hptt_code_gen_trans WORKING_DIR)
  message("-- Generating transpose library template explicit instantation. "
    "Target directory: " ${HPTT_CODE_GEN_TARGET_DIR})
  execute_process(
    COMMAND ${HPTT_PYTHON_EXEC} ${HPTT_CODE_GEN_TRANS_SCRIPT}
    ${HPTT_CODE_GEN_TARGET_DIR_ARG} ${HPTT_CODE_GEN_TARGET_DIR}
    ${HPTT_CODE_GEN_TRANS_ORDER_MIN_ARG} ${HPTT_CODE_GEN_TRANS_ORDER_MIN}
    ${HPTT_CODE_GEN_TRANS_ORDER_MAX_ARG} ${HPTT_CODE_GEN_TRANS_ORDER_MAX}
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

macro(hptt_code_gen_all WORKING_DIR)
  hptt_code_gen_trans(${WORKING_DIR})
endmacro()
