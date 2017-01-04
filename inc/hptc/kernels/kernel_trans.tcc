#pragma once
#ifndef HPTC_KERNELS_KERNEL_TRANS_TCC_
#define HPTC_KERNELS_KERNEL_TRANS_TCC_

template <typename FloatType>
struct KernelTransScalar<FloatType, CoefUsage::USE_NONE> {
  INLINE void operator()(const FloatType * RESTRICT input_data,
      FloatType * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride, DeducedFloatType<FloatType> alpha,
      DeducedFloatType<FloatType> beta) {
    using Deduced = DeducedFloatType<FloatType>;
    constexpr auto inner_offset = sizeof(FloatType) / sizeof(Deduced);
    auto input_ptr = reinterpret_cast<Deduced *>(output_data);
    auto output_ptr = reinterpret_cast<Deduced *>(output_data);
    for (TensorIdx idx = 0; idx < inner_offset; ++idx)
      output_ptr[idx] = input_ptr[idx];
  }
};


template <typename FloatType>
struct KernelTransScalar<FloatType, CoefUsage::USE_ALPHA> {
  INLINE void operator()(const FloatType * RESTRICT input_data,
      FloatType * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride, DeducedFloatType<FloatType> alpha,
      DeducedFloatType<FloatType> beta) {
    using Deduced = DeducedFloatType<FloatType>;
    constexpr auto inner_offset = sizeof(FloatType) / sizeof(Deduced);
    auto input_ptr = reinterpret_cast<Deduced *>(output_data);
    auto output_ptr = reinterpret_cast<Deduced *>(output_data);
    for (TensorIdx idx = 0; idx < inner_offset; ++idx)
      output_ptr[idx] = alpha * input_ptr[idx];
  }
};


template <typename FloatType>
struct KernelTransScalar<FloatType, CoefUsage::USE_BETA> {
  INLINE void operator()(const FloatType * RESTRICT input_data,
      FloatType * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride, DeducedFloatType<FloatType> alpha,
      DeducedFloatType<FloatType> beta) {
    using Deduced = DeducedFloatType<FloatType>;
    constexpr auto inner_offset = sizeof(FloatType) / sizeof(Deduced);
    auto input_ptr = reinterpret_cast<Deduced *>(output_data);
    auto output_ptr = reinterpret_cast<Deduced *>(output_data);
    for (TensorIdx idx = 0; idx < inner_offset; ++idx)
      output_ptr[idx] = input_ptr[idx] + beta * output_ptr[idx];
  }
};


template <typename FloatType>
struct KernelTransScalar<FloatType, CoefUsage::USE_BOTH> {
  INLINE void operator()(const FloatType * RESTRICT input_data,
      FloatType * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride, DeducedFloatType<FloatType> alpha,
      DeducedFloatType<FloatType> beta) {
    using Deduced = DeducedFloatType<FloatType>;
    constexpr auto inner_offset = sizeof(FloatType) / sizeof(Deduced);
    auto input_ptr = reinterpret_cast<Deduced *>(output_data);
    auto output_ptr = reinterpret_cast<Deduced *>(output_data);
    for (TensorIdx idx = 0; idx < inner_offset; ++idx)
      output_ptr[idx] = alpha * input_ptr[idx] + beta * output_ptr[idx];
  }
};

#endif // HPTC_KERNELS_KERNEL_TRANS_TCC_
