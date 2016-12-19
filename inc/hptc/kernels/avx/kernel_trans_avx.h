#pragma once
#ifndef HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_H_
#define HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_H_

#include <immintrin.h>

#include <type_traits>

#include <hptc/types.h>
#include <hptc/param/parameter_trans.h>


namespace hptc {

/*
 * Register type deducing utilities
 */
template <typename FloatType>
struct RegDeducer {
};

template <>
struct RegDeducer<float> {
  using type = __m256;
};

template <>
struct RegDeducer<double> {
  using type = __m256d;
};

template <>
struct RegDeducer<FloatComplex> {
  using type = __m256;
};

template <>
struct RegDeducer<DoubleComplex> {
  using type = __m256d;
};

template <typename FloatType>
using DeducedRegType = typename RegDeducer<FloatType>::type;


template <typename DeducedFloat>
INLINE DeducedRegType<DeducedFloat> reg_coef(DeducedFloat coef);


enum class KernelType : bool {
  KERNEL_FULL = true,
  KERNEL_HALF = false
};


template <typename FloatType,
          KernelType TYPE>
struct KernelTransAvxBase {
  INLINE GenNumType get_reg_num();
};


template <typename FloatType,
          CoefUsage USAGE,
          KernelType TYPE = KernelType::KERNEL_FULL>
struct KernelTransAvx final : public KernelTransAvxBase<FloatType, TYPE> {
};


template <CoefUsage USAGE>
struct KernelTransAvx<float, USAGE, KernelType::KERNEL_FULL> final
    : public KernelTransAvxBase<float, KernelType::KERNEL_FULL> {
  INLINE void operator()(const float * RESTRICT input_data,
      float * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride, __m256 &reg_alpha, __m256 &reg_beta);
};


template <CoefUsage USAGE>
struct KernelTransAvx<double, USAGE, KernelType::KERNEL_FULL> final
    : public KernelTransAvxBase<double, KernelType::KERNEL_FULL> {
  INLINE void operator()(const double * RESTRICT input_data,
      double * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride, __m256d &reg_alpha, __m256d &reg_beta);
};


template <CoefUsage USAGE>
struct KernelTransAvx<FloatComplex, USAGE, KernelType::KERNEL_FULL> final
    : public KernelTransAvxBase<FloatComplex, KernelType::KERNEL_FULL> {
  INLINE void operator()(const FloatComplex * RESTRICT input_data,
      FloatComplex * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride, __m256 &reg_alpha, __m256 &reg_beta);
};


template <CoefUsage USAGE>
struct KernelTransAvx<DoubleComplex, USAGE, KernelType::KERNEL_FULL> final
    : public KernelTransAvxBase<DoubleComplex, KernelType::KERNEL_FULL> {
  INLINE void operator()(const DoubleComplex * RESTRICT input_data,
      DoubleComplex * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride, __m256d &reg_alpha, __m256d &reg_beta);
};


template <CoefUsage USAGE>
struct KernelTransAvx<float, USAGE, KernelType::KERNEL_HALF> final
    : public KernelTransAvxBase<float, KernelType::KERNEL_HALF> {
  INLINE void operator()(const float * RESTRICT input_data,
      float * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride, __m256 &reg_alpha, __m256 &reg_beta);
};


template <CoefUsage USAGE>
struct KernelTransAvx<double, USAGE, KernelType::KERNEL_HALF> final
    : public KernelTransAvxBase<double, KernelType::KERNEL_HALF> {
  INLINE void operator()(const double * RESTRICT input_data,
      double * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride, __m256d &reg_alpha, __m256d &reg_beta);
};


template <CoefUsage USAGE>
struct KernelTransAvx<FloatComplex, USAGE, KernelType::KERNEL_HALF> final
    : public KernelTransAvxBase<FloatComplex, KernelType::KERNEL_HALF> {
  INLINE void operator()(const FloatComplex * RESTRICT input_data,
      FloatComplex * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride, __m256 &reg_alpha, __m256 &reg_beta);
};


template <CoefUsage USAGE>
struct KernelTransAvx<DoubleComplex, USAGE, KernelType::KERNEL_HALF> final
    : public KernelTransAvxBase<DoubleComplex, KernelType::KERNEL_HALF> {
  INLINE void operator()(const DoubleComplex * RESTRICT input_data,
      DoubleComplex * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride, __m256d &reg_alpha, __m256d &reg_beta);
};


/*
 * Import implementation
 */
#include "kernel_trans_avx.tcc"

}

#endif // HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_H_
