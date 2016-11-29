#pragma once
#ifndef HPTC_KERNELS_AVX_INTRIN_AVX_TCC_
#define HPTC_KERNELS_AVX_INTRIN_AVX_TCC_

template <GenNumType GEN_NUM,
          typename Intrin,
          typename... Args>
INLINE void intrin_tiler(GenCounter<GEN_NUM>, Intrin intrinsic, Args... args) {
  intrin_tiler(GenCounter<GEN_NUM - 1>(), intrinsic, GEN_NUM - 1, args...);
  intrinsic(GEN_NUM, args...);
}


template <typename Intrin,
          typename... Args>
INLINE void intrin_tiler(GenCounter<0>, Intrin intrinsic, Args... args) {
  intrinsic(0, args...);
}


template <>
INLINE void intrin_load(GenNumType intrin_idx, const float * RESTRICT data,
    TensorIdx offset, DeducedRegType<float> * const reg) {
  reg[intrin_idx] = _mm256_load_ps(data + intrin_idx * offset);
}


template <>
INLINE void intrin_load(GenNumType intrin_idx, const double * RESTRICT data,
    TensorIdx offset, DeducedRegType<double> * const reg) {
  reg[intrin_idx] = _mm256_load_pd(data + intrin_idx * offset);
}


template <>
INLINE void intrin_store(GenNumType intrin_idx, float * RESTRICT data,
    TensorIdx offset, const DeducedRegType<float> * const reg) {
  _mm256_store_ps(data + intrin_idx * offset, reg[intrin_idx]);
}


template <>
INLINE void intrin_store(GenNumType intrin_idx, double * RESTRICT data,
    TensorIdx offset, const DeducedRegType<double> * const reg) {
  _mm256_store_pd(data + intrin_idx * offset, reg[intrin_idx]);
}


template <>
INLINE void intrin_set1(float val, DeducedRegType<float> *reg) {
  *reg = _mm256_set1_ps(val);
}


template <>
INLINE void intrin_set1(double val, DeducedRegType<double> *reg) {
  *reg = _mm256_set1_pd(val);
}


template <>
INLINE void intrin_mul(GenNumType intrin_idx,
    DeducedRegType<float> * const reg_scaled, DeducedRegType<float> reg_coef) {
  reg_scaled[intrin_idx] = _mm256_mul_ps(reg_scaled[intrin_idx], reg_coef);
}


template <>
INLINE void intrin_mul(GenNumType intrin_idx,
    DeducedRegType<double> * const reg_scaled,
    DeducedRegType<double> reg_coef) {
  reg_scaled[intrin_idx] = _mm256_mul_pd(reg_scaled[intrin_idx], reg_coef);
}


template <>
INLINE void intrin_add(GenNumType intrin_idx,
    DeducedRegType<float> * const reg_output,
    const DeducedRegType<float> * const reg_input) {
  reg_output[intrin_idx] = _mm256_add_ps(reg_output[intrin_idx],
      reg_input[intrin_idx]);
}


template <>
INLINE void intrin_add(GenNumType intrin_idx,
    DeducedRegType<double> * const reg_output,
    const DeducedRegType<double> * const reg_input) {
  reg_output[intrin_idx] = _mm256_add_pd(reg_output[intrin_idx],
      reg_input[intrin_idx]);
}

#endif // HPTC_KERNELS_AVX_INTRIN_AVX_TCC_
