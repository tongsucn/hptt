#pragma once
#ifndef HPTT_ARCH_COMMON_COMMON_IMPL_H_
#define HPTT_ARCH_COMMON_COMMON_IMPL_H_

#include <hptt/types.h>
#include <hptt/arch/compat.h>


namespace hptt {

template <typename FloatType,
          TensorUInt WIDTH,
          bool UPDATE_OUT>
HPTT_INL void common_trans_impl(const FloatType * RESTRICT data_in,
    FloatType * RESTRICT data_out, const TensorIdx stride_in_outld,
    const TensorIdx stride_out_inld, const DeducedFloatType<FloatType> &alpha,
    const DeducedFloatType<FloatType> &beta) {
  if (UPDATE_OUT)
#pragma omp simd collapse(2)
    for (TensorUInt idx_in_outld = 0; idx_in_outld < WIDTH; ++idx_in_outld) {
      for (TensorUInt idx_in_inld = 0; idx_in_inld < WIDTH; ++idx_in_inld) {
        const auto idx_in = idx_in_inld + idx_in_outld * stride_in_outld,
            idx_out = idx_in_outld + idx_in_inld * stride_out_inld;
        data_out[idx_out] = alpha * data_in[idx_in] + beta * data_out[idx_out];
      }
    }
  else
#pragma omp simd collapse(2)
    for (TensorUInt idx_in_outld = 0; idx_in_outld < WIDTH; ++idx_in_outld) {
      for (TensorUInt idx_in_inld = 0; idx_in_inld < WIDTH; ++idx_in_inld) {
        const auto idx_in = idx_in_inld + idx_in_outld * stride_in_outld,
            idx_out = idx_in_outld + idx_in_inld * stride_out_inld;
        data_out[idx_out] = alpha * data_in[idx_in];
      }
    }
}


template <typename FloatType,
          bool UPDATE_OUT>
HPTT_INL void common_trans_linear_impl(const FloatType * RESTRICT data_in,
    FloatType * RESTRICT data_out, const TensorIdx size_trans,
    const TensorIdx, const DeducedFloatType<FloatType> &alpha,
    const DeducedFloatType<FloatType> &beta) {
  if (UPDATE_OUT)
#pragma omp simd
    for (TensorIdx idx = 0; idx < size_trans; ++idx)
      data_out[idx] = alpha * data_in[idx] + beta * data_out[idx];
  else
#pragma omp simd
    for (TensorIdx idx = 0; idx < size_trans; ++idx)
      data_out[idx] = alpha * data_in[idx];
}

}

#endif // HPTT_ARCH_COMMON_COMMON_IMPL_H_
