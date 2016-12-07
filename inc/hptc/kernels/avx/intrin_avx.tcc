#pragma once
#ifndef HPTC_KERNELS_AVX_INTRIN_AVX_TCC_
#define HPTC_KERNELS_AVX_INTRIN_AVX_TCC_

template <>
inline void intrin_avx_load<float>(GenNumType intrin_idx,
    const float * RESTRICT data, TensorIdx offset,
    DeducedRegType<float> reg[]) {
  reg[intrin_idx] = _mm256_loadu_ps(data + intrin_idx * offset);
}


template <>
inline void intrin_avx_load<double>(GenNumType intrin_idx,
    const double * RESTRICT data, TensorIdx offset,
    DeducedRegType<double> reg[]) {
  reg[intrin_idx] = _mm256_loadu_pd(data + intrin_idx * offset);
}


template <>
inline void intrin_avx_store<float>(GenNumType intrin_idx,
    float * RESTRICT data, TensorIdx offset,
    const DeducedRegType<float> reg[]) {
  _mm256_storeu_ps(data + intrin_idx * offset, reg[intrin_idx]);
}


template <>
inline void intrin_avx_store<double>(GenNumType intrin_idx,
    double * RESTRICT data, TensorIdx offset,
    const DeducedRegType<double> reg[]) {
  _mm256_storeu_pd(data + intrin_idx * offset, reg[intrin_idx]);
}


template <>
inline void intrin_avx_set1<float>(float val, DeducedRegType<float> *reg) {
  *reg = _mm256_set1_ps(val);
}


template <>
inline void intrin_avx_set1<double>(double val, DeducedRegType<double> *reg) {
  *reg = _mm256_set1_pd(val);
}


template <>
inline void intrin_avx_mul<float>(GenNumType intrin_idx,
    DeducedRegType<float> reg_scaled[], DeducedRegType<float> reg_coef) {
  reg_scaled[intrin_idx] = _mm256_mul_ps(reg_scaled[intrin_idx], reg_coef);
}


template <>
inline void intrin_avx_mul<double>(GenNumType intrin_idx,
    DeducedRegType<double> reg_scaled[], DeducedRegType<double> reg_coef) {
  reg_scaled[intrin_idx] = _mm256_mul_pd(reg_scaled[intrin_idx], reg_coef);
}


template <>
inline void intrin_avx_add<float>(GenNumType intrin_idx,
    DeducedRegType<float> reg_output[],
    const DeducedRegType<float> reg_input[]) {
  reg_output[intrin_idx] = _mm256_add_ps(reg_output[intrin_idx],
      reg_input[intrin_idx]);
}


template <>
inline void intrin_avx_add<double>(GenNumType intrin_idx,
    DeducedRegType<double> reg_output[],
    const DeducedRegType<double> reg_input[]) {
  reg_output[intrin_idx] = _mm256_add_pd(reg_output[intrin_idx],
      reg_input[intrin_idx]);
}


template <>
inline void intrin_avx_trans<float>(DeducedRegType<float> reg_input[]) {
  DeducedRegType<float> reg[16];
  reg[0] = _mm256_unpacklo_ps(reg_input[0], reg_input[1]);
  reg[1] = _mm256_unpackhi_ps(reg_input[0], reg_input[1]);
  reg[2] = _mm256_unpacklo_ps(reg_input[2], reg_input[3]);
  reg[3] = _mm256_unpackhi_ps(reg_input[2], reg_input[3]);
  reg[4] = _mm256_unpacklo_ps(reg_input[4], reg_input[5]);
  reg[5] = _mm256_unpackhi_ps(reg_input[4], reg_input[5]);
  reg[6] = _mm256_unpacklo_ps(reg_input[6], reg_input[7]);
  reg[7] = _mm256_unpackhi_ps(reg_input[6], reg_input[7]);

  reg[8] = _mm256_shuffle_ps(reg[0], reg[2], 0x44);
  reg[9] = _mm256_shuffle_ps(reg[0], reg[2], 0xEE);
  reg[10] = _mm256_shuffle_ps(reg[1], reg[3], 0x44);
  reg[11] = _mm256_shuffle_ps(reg[1], reg[3], 0xEE);
  reg[12] = _mm256_shuffle_ps(reg[4], reg[6], 0x44);
  reg[13] = _mm256_shuffle_ps(reg[4], reg[6], 0xEE);
  reg[14] = _mm256_shuffle_ps(reg[5], reg[7], 0x44);
  reg[15] = _mm256_shuffle_ps(reg[5], reg[7], 0xEE);

  reg_input[0] = _mm256_permute2f128_ps(reg[12], reg[8], 0x2);
  reg_input[1] = _mm256_permute2f128_ps(reg[13], reg[9], 0x2);
  reg_input[2] = _mm256_permute2f128_ps(reg[14], reg[10], 0x2);
  reg_input[3] = _mm256_permute2f128_ps(reg[15], reg[11], 0x2);
  reg_input[4] = _mm256_permute2f128_ps(reg[12], reg[8], 0x13);
  reg_input[5] = _mm256_permute2f128_ps(reg[13], reg[9], 0x13);
  reg_input[6] = _mm256_permute2f128_ps(reg[14], reg[10], 0x13);
  reg_input[7] = _mm256_permute2f128_ps(reg[15], reg[11], 0x13);
}


template <>
inline void intrin_avx_trans<double>(DeducedRegType<double> reg_input[]) {
  DeducedRegType<double> reg[4];
  reg[0] = _mm256_shuffle_pd(reg_input[0], reg_input[1], 0x3);
  reg[1] = _mm256_shuffle_pd(reg_input[0], reg_input[1], 0xC);
  reg[2] = _mm256_shuffle_pd(reg_input[2], reg_input[3], 0x3);
  reg[3] = _mm256_shuffle_pd(reg_input[2], reg_input[3], 0xC);
  reg_input[0] = _mm256_permute2f128_pd(reg[3], reg[1], 0x2);
  reg_input[1] = _mm256_permute2f128_pd(reg[2], reg[0], 0x2);
  reg_input[2] = _mm256_permute2f128_pd(reg[2], reg[0], 0x13);
  reg_input[3] = _mm256_permute2f128_pd(reg[3], reg[1], 0x13);
}


template <>
inline void intrin_avx_trans<FloatComplex>(
    DeducedRegType<FloatComplex> reg_input[]) {
  DeducedRegType<FloatComplex> reg[4];
  reg[0] = _mm256_shuffle_ps(reg_input[0], reg_input[1], 0x44);
  reg[1] = _mm256_shuffle_ps(reg_input[0], reg_input[1], 0xEE);
  reg[2] = _mm256_shuffle_ps(reg_input[2], reg_input[3], 0x44);
  reg[3] = _mm256_shuffle_ps(reg_input[2], reg_input[3], 0xEE);
  reg_input[0] = _mm256_permute2f128_ps(reg[2], reg[0], 0x2);
  reg_input[1] = _mm256_permute2f128_ps(reg[3], reg[1], 0x2);
  reg_input[2] = _mm256_permute2f128_ps(reg[2], reg[0], 0x13);
  reg_input[3] = _mm256_permute2f128_ps(reg[3], reg[1], 0x13);
}


template <>
inline void intrin_avx_trans<DoubleComplex>(
    DeducedRegType<DoubleComplex> reg_input[]) {
  __m256d t0;
  DeducedRegType<DoubleComplex> reg[2];
  reg[0] = _mm256_permute2f128_pd(reg_input[1], reg_input[0], 0x2);
  reg[1] = _mm256_permute2f128_pd(reg_input[1], reg_input[0], 0x13);
  reg_input[0] = reg[0];
  reg_input[1] = reg[1];
}

#endif // HPTC_KERNELS_AVX_INTRIN_AVX_TCC_
