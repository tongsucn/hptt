#pragma once
#ifndef HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_H_
#define HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_H_

#include <immintrin.h>

#include <type_traits>
#include <iostream>

#include <hptc/types.h>
#include <hptc/kernels/kernel_trans_base.h>


namespace hptc {

template <typename FloatType,
          CoefUsage USAGE>
class KernelTransAvxImpl final : public KernelTransBase<FloatType> {
public:
  KernelTransAvxImpl(DeducedFloatType<FloatType> alpha,
      DeducedFloatType<FloatType> beta);

  KernelTransAvxImpl(const KernelTransAvxImpl<FloatType, USAGE> &kernel)
      = delete;
  KernelTransAvxImpl &operator=(const KernelTransAvxImpl &kernel) = delete;

  virtual ~KernelTransAvxImpl() = default;

  virtual INLINE void operator()(const float * RESTRICT input_data,
      float * RESTRICT output_data, TensorIdx input_stride,
      TensorIdx output_stride) final;

  virtual INLINE void operator()(const double * RESTRICT input_data,
      double * RESTRICT output_data, TensorIdx input_stride,
      TensorIdx output_stride) final;

  virtual INLINE void operator()(const FloatComplex * RESTRICT input_data,
      FloatComplex * RESTRICT output_data, TensorIdx input_stride,
      TensorIdx output_stride) final;

  virtual INLINE void operator()(const DoubleComplex * RESTRICT input_data,
      DoubleComplex * RESTRICT output_data, TensorIdx input_stride,
      TensorIdx output_stride) final;

  INLINE GenNumType get_reg_num() final;

protected:
  __m256 reg_alpha, reg_beta;
  __m256d regd_alpha, regd_beta;
};


/*
 * Import implementation
 */
#include "kernel_trans_avx.tcc"

}

#endif // HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_H_
