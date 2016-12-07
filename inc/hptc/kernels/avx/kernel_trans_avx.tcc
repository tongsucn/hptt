#pragma once
#ifndef HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_TCC_
#define HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_TCC_

template <typename FloatType,
          GenNumType REG_NUM>
KernelTransAvxBase<FloatType, REG_NUM>::KernelTransAvxBase()
    : offset_scale(sizeof(FloatType) / sizeof(DeducedFloatType<FloatType>)) {
  this->out_reg_arr_ptr = this->out_reg_arr;
}


template <typename FloatType,
          GenNumType REG_NUM>
INLINE GenNumType KernelTransAvxBase<FloatType, REG_NUM>::get_reg_num() {
  return REG_NUM;
}


template <typename FloatType,
          GenNumType REG_NUM>
INLINE void KernelTransAvxBase<FloatType, REG_NUM>::in_reg_trans(
    const FloatType * RESTRICT input_data, TensorIdx input_offset) {
  // Load input data into registers
  intrin_tiler(GenCounter<static_cast<GenNumType>(REG_NUM - 1)>(),
      intrin_avx_load<DeducedFloatType<FloatType>>,
      reinterpret_cast<const DeducedFloatType<FloatType> *>(input_data),
      input_offset * this->offset_scale, this->in_reg_arr);

  // Execute transpose
  intrin_avx_trans<FloatType>(this->in_reg_arr);
}


template <typename FloatType,
          GenNumType REG_NUM>
INLINE void KernelTransAvxBase<FloatType, REG_NUM>::rescale_input(
    DeducedFloatType<FloatType> alpha) {
  // No implementation needed here
}


template <typename FloatType,
          GenNumType REG_NUM>
INLINE void KernelTransAvxBase<FloatType, REG_NUM>::update_output(
    FloatType * RESTRICT output_data, TensorIdx output_offset,
    DeducedFloatType<FloatType> beta) {
  // Need only copy data from input register to output register
  this->out_reg_arr_ptr = this->in_reg_arr;
}


template <typename FloatType,
          GenNumType REG_NUM>
INLINE void KernelTransAvxBase<FloatType, REG_NUM>::write_back(
    FloatType * RESTRICT output_data, TensorIdx output_offset) {
  // Store reuslts
  intrin_tiler(GenCounter<static_cast<GenNumType>(REG_NUM - 1)>(),
      intrin_avx_store<DeducedFloatType<FloatType>>,
      reinterpret_cast<DeducedFloatType<FloatType> *>(output_data),
      output_offset * this->offset_scale, this->out_reg_arr_ptr);
}


template <typename FloatType,
          GenNumType REG_NUM>
INLINE void
KernelTransAvxImpl<FloatType, CoefUsage::USE_ALPHA, REG_NUM>::rescale_input(
    DeducedFloatType<FloatType> alpha) {
  // Rescale with alpha
  DeducedRegType<FloatType> reg_alpha;
  intrin_avx_set1<DeducedFloatType<FloatType>>(alpha, &reg_alpha);
  intrin_tiler(GenCounter<static_cast<GenNumType>(REG_NUM - 1)>(),
      intrin_avx_mul<DeducedFloatType<FloatType>>, this->in_reg_arr,
      reg_alpha);
}


template <typename FloatType,
          GenNumType REG_NUM>
INLINE void
KernelTransAvxImpl<FloatType, CoefUsage::USE_BETA, REG_NUM>::update_output(
    FloatType * RESTRICT output_data, TensorIdx output_offset,
    DeducedFloatType<FloatType> beta) {
  // Load output data into register
  intrin_tiler(GenCounter<static_cast<GenNumType>(REG_NUM - 1)>(),
      intrin_avx_load<DeducedFloatType<FloatType>>,
      reinterpret_cast<const DeducedFloatType<FloatType> *>(output_data),
      output_offset * this->offset_scale, this->out_reg_arr);

  // Rescale with beta
  DeducedRegType<FloatType> reg_beta;
  intrin_avx_set1<DeducedFloatType<FloatType>>(beta, &reg_beta);
  intrin_tiler(GenCounter<static_cast<GenNumType>(REG_NUM - 1)>(),
      intrin_avx_mul<DeducedFloatType<FloatType>>, this->out_reg_arr, reg_beta);

  // Update output data
  intrin_tiler(GenCounter<static_cast<GenNumType>(REG_NUM - 1)>(),
      intrin_avx_add<DeducedFloatType<FloatType>>, this->out_reg_arr,
      this->in_reg_arr);
}


template <typename FloatType,
          GenNumType REG_NUM>
INLINE void
KernelTransAvxImpl<FloatType, CoefUsage::USE_BOTH, REG_NUM>::rescale_input(
    DeducedFloatType<FloatType> alpha) {
  // Rescale with alpha
  DeducedRegType<FloatType> reg_alpha;
  intrin_avx_set1<DeducedFloatType<FloatType>>(alpha, &reg_alpha);
  intrin_tiler(GenCounter<static_cast<GenNumType>(REG_NUM - 1)>(),
      intrin_avx_mul<DeducedFloatType<FloatType>>, this->in_reg_arr,
      reg_alpha);
}


template <typename FloatType,
          GenNumType REG_NUM>
INLINE void
KernelTransAvxImpl<FloatType, CoefUsage::USE_BOTH, REG_NUM>::update_output(
    FloatType * RESTRICT output_data, TensorIdx output_offset,
    DeducedFloatType<FloatType> beta) {
  // Load output data into register
  intrin_tiler(GenCounter<static_cast<GenNumType>(REG_NUM - 1)>(),
      intrin_avx_load<DeducedFloatType<FloatType>>,
      reinterpret_cast<const DeducedFloatType<FloatType> *>(output_data),
      output_offset * this->offset_scale, this->out_reg_arr);

  // Rescale with beta
  DeducedRegType<FloatType> reg_beta;
  intrin_avx_set1<DeducedFloatType<FloatType>>(beta, &reg_beta);
  intrin_tiler(GenCounter<static_cast<GenNumType>(REG_NUM - 1)>(),
      intrin_avx_mul<DeducedFloatType<FloatType>>, this->out_reg_arr, reg_beta);

  // Update output data
  intrin_tiler(GenCounter<static_cast<GenNumType>(REG_NUM - 1)>(),
      intrin_avx_add<DeducedFloatType<FloatType>>, this->out_reg_arr,
      this->in_reg_arr);
}

#endif // HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_TCC_
