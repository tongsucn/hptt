#pragma once
#ifndef HPTC_KERNELS_KERNEL_TRANS_TCC_
#define HPTC_KERNELS_KERNEL_TRANS_TCC_

template <typename FloatType,
          CoefUsage USAGE>
INLINE void kernel_trans_full(const FloatType * RESTRICT input_data,
    FloatType * RESTRICT output_data, const TensorIdx input_stride,
    const TensorIdx output_stride, DeducedRegType<FloatType> &reg_alpha,
    DeducedRegType<FloatType> &reg_beta) {
  // AVX implementation
  kernel_trans_full_avx(input_data, output_data, input_stride, output_stride,
      reg_alpha, reg_beta);
}


template <typename FloatType,
          CoefUsage USAGE>
INLINE void kernel_trans_half(const FloatType * RESTRICT input_data,
    FloatType * RESTRICT output_data, const TensorIdx input_stride,
    const TensorIdx output_stride, DeducedRegType<FloatType> &reg_alpha,
    DeducedRegType<FloatType> &reg_beta) {
  // AVX implementation
  kernel_trans_half_avx(input_data, output_data, input_stride, output_stride,
      reg_alpha, reg_beta);
}

#endif // HPTC_KERNELS_KERNEL_TRANS_TCC_
