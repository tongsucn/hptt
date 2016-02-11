# ----------------------------------------------------------------------------
# Initialize library setting
# ----------------------------------------------------------------------------
macro(hptt_init_lib_setting)
  set(HPTT_LIB_NAME "hptt")
  set(HPTT_LIB_STATIC_NAME "hptt_static")
  set(HPTT_AVX2_LIB_NAME "hptt_avx2")
  set(HPTT_AVX_LIB_NAME "hptt_avx")
  set(HPTT_ARM_LIB_NAME "hptt_arm")
  set(HPTT_IBM_LIB_NAME "hptt_ibm")
  set(HPTT_COMMON_LIB_NAME "hptt_common")
  set(HPTT_BENCHMARK_LIB_NAME "hptt_benchmark")
endmacro()
