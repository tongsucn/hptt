# ----------------------------------------------------------------------------
# Code generation
# ----------------------------------------------------------------------------
string(CONCAT HPTC_CODE_GEN_SCRIPT_DIR ${CMAKE_CURRENT_SOURCE_DIR}
  "/cmake/pylib")
string(CONCAT HPTC_CODE_GEN_TRANS_SCRIPT ${HPTC_CODE_GEN_SCRIPT_DIR}
  "/code_gen_trans.py")

# ----------------------------------------------------------------------------
# Code generation
# ----------------------------------------------------------------------------
macro(hptc_code_gen_trans GEN_SCRIPT WORKING_DIR)
  message(${HPTC_MSG_CODE_GEN_TRANS})
  execute_process(
    COMMAND ${HPTC_PYTHON_EXEC} ${GEN_SCRIPT}
    ${HPTC_CODE_GEN_TARGET_DIR_ARG} ${HPTC_CODE_GEN_TARGET_DIR}
    ${HPTC_CODE_GEN_TRANS_ORDER_MIN_ARG} ${HPTC_CODE_GEN_TRANS_ORDER_MIN}
    ${HPTC_CODE_GEN_TRANS_ORDER_MAX_ARG} ${HPTC_CODE_GEN_TRANS_ORDER_MAX}
    ${HPTC_CODE_GEN_TRANS_DTYPE_ARG} ${HPTC_CODE_GEN_TRANS_DTYPE}
    ${HPTC_CODE_GEN_TRANS_COEF_ARG} ${HPTC_CODE_GEN_TRANS_COEF}
    WORKING_DIRECTORY ${WORKING_DIR}
    OUTPUT_VARIABLE CODE_GEN_STDOUT
    ERROR_VARIABLE CODE_GEN_STDERR
    RESULT_VARIABLE RET_VAL
    )

  if (NOT RET_VAL EQUAL 0)
    message(FATAL_ERROR ${HPTC_MSG_FATAL_CODE_GEN_TRANS})
  endif ()
endmacro()

macro(hptc_code_gen_all WORKING_DIR)
  hptc_code_gen_trans(${HPTC_CODE_GEN_TRANS_SCRIPT} ${WORKING_DIR})
endmacro()