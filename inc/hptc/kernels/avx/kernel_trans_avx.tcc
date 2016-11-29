#pragma once
#ifndef HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_TCC_
#define HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_TCC_

template <typename FloatType
          uint32_t REG_NUM>
INLINE void KernelTransAvxBase<FloatType, REG_NUM>::in_reg_trans(
    const FloatType * RESTRICT input_data, TensorIdx input_offset) {
  // Load input data into registers
  intrin_tiler(GenCounter<static_cast<GenNumType>(REG_NUM - 1)>(),
      intrin_load<DeducedFloatType<FloatType>>,
      reinterpret_cast<const DeducedFloatType<FloatType> *>(input_data),
      input_offset, this->in_reg_arr);
}


template <typename FloatType
          uint32_t REG_NUM>
INLINE void KernelTransAvxBase<FloatType, REG_NUM>::write_back(
    FloatType * RESTRICT output_data, TensorIdx output_offset) {
  // Store reuslts
  intrin_tiler(GenCounter<static_cast<GenNumType>(REG_NUM - 1)>(),
      intrin_store<DeducedFloatType<FloatType>>,
      reinterpret_cast<const DeducedFloatType<FloatType> *(output_data),
      output_offset, this->out_reg_arr);
}


template <typename FloatType
          uint32_t REG_NUM>
INLINE void KernelTransAvxBase<FloatType, REG_NUM>::rescale_input(
    DeducedFloatType<FloatType> alpha) {
  // No implementation here.
}


template <typename FloatType
          uint32_t REG_NUM>
INLINE void KernelTransAvxBase<FloatType, REG_NUM>::update_output(
    FloatType * RESTRICT output_data, TensorIdx output_offset,
    DeducedFloatType<FloatType> beta) {
  // No implementation here.
}


template <typename FloatType
          uint32_t REG_NUM>
INLINE void
KernelTransAvx<FloatType, CoefUsage::USE_ALPHA, REG_NUM>::rescale_input(
    DeducedFloatType<FloatType> alpha) {
  // Rescale with alpha
  DeducedRegType<FloatType> reg_alpha;
  intrin_set1<DeducedFloatType<FloatType>>(alpha, &reg_alpha);
  intrin_tiler(GenCounter<static_cast<GenNumType>(REG_NUM - 1)>(),
      intrin_mul<DeducedFloatType<FloatType>>, this->int_reg_arr, reg_alpha);
}


template <typename FloatType
          uint32_t REG_NUM>
INLINE void
KernelTransAvx<FloatType, CoefUsage::USE_BETA, REG_NUM>::update_output(
    FloatType * RESTRICT output_data, TensorIdx output_offset,
    DeducedFloatType<FloatType> beta) {
  // Load output data into register
  intrin_tiler(GenCounter<static_cast<GenNumType>(REG_NUM - 1)>(),
      intrin_load<DeducedFloatType<FloatType>>,
      reinterpret_cast<const DeducedFloatType<FloatType> *>(output_data),
      output_offset, this->out_reg_arr);

  // Rescale with beta
  DeducedRegType<FloatType> reg_beta;
  intrin_set1<DeducedFloatType<FloatType>>(beta, &reg_beta);
  intrin_tiler(GenCounter<static_cast<GenNumType>(REG_NUM - 1)>(),
      intrin_mul<DeducedFloatType<FloatType>>, this->out_reg_arr, reg_beta);

  // Update output data
  intrin_tiler(GenCounter<static_cast<GenNumType>(REG_NUM - 1)>(),
      intrin_add<DeducedFloatType<FloatType>>, this->out_reg_arr,
      this->in_reg_arr);
}


template <typename FloatType
          uint32_t REG_NUM>
INLINE void
KernelTransAvx<FloatType, CoefUsage::USE_BOTH, REG_NUM>::rescale_input(
    DeducedFloatType<FloatType> alpha) {
  // Rescale with alpha
  DeducedRegType<FloatType> reg_alpha;
  intrin_set1<DeducedFloatType<FloatType>>(alpha, &reg_alpha);
  intrin_tiler(GenCounter<static_cast<GenNumType>(REG_NUM - 1)>(),
      intrin_mul<DeducedFloatType<FloatType>>, this->int_reg_arr, reg_alpha);
}


template <typename FloatType
          uint32_t REG_NUM>
INLINE void
KernelTransAvx<FloatType, CoefUsage::USE_BOTH, REG_NUM>::update_output(
    FloatType * RESTRICT output_data, TensorIdx output_offset,
    DeducedFloatType<FloatType> beta) {
  // Load output data into register
  intrin_tiler(GenCounter<static_cast<GenNumType>(REG_NUM - 1)>(),
      intrin_load<DeducedFloatType<FloatType>>,
      reinterpret_cast<const DeducedFloatType<FloatType> *>(output_data),
      output_offset, this->out_reg_arr);

  // Rescale with beta
  DeducedRegType<FloatType> reg_beta;
  intrin_set1<DeducedFloatType<FloatType>>(beta, &reg_beta);
  intrin_tiler(GenCounter<static_cast<GenNumType>(REG_NUM - 1)>(),
      intrin_mul<DeducedFloatType<FloatType>>, this->out_reg_arr, reg_beta);

  // Update output data
  intrin_tiler(GenCounter<static_cast<GenNumType>(REG_NUM - 1)>(),
      intrin_add<DeducedFloatType<FloatType>>, this->out_reg_arr,
      this->in_reg_arr);
}

#endif // HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_TCC_
