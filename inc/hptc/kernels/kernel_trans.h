#pragma once
#ifndef HPTC_KERNELS_KERNEL_TRANS_H_
#define HPTC_KERNELS_KERNEL_TRANS_H_

#include <hptc/param/parameter_trans.h>
#include <hptc/kernels/avx/kernel_trans_avx.h>


namespace hptc {

template <typename FloatType,
          CoefUsage USAGE>
INLINE void kernel_trans_full(const FloatType * RESTRICT input_data,
    FloatType * RESTRICT output_data, const TensorIdx input_stride,
    const TensorIdx output_stride, DeducedRegType<FloatType> &reg_alpha,
    DeducedRegType<FloatType> &reg_beta);


template <typename FloatType,
          CoefUsage USAGE>
INLINE void kernel_trans_half(const FloatType * RESTRICT input_data,
    FloatType * RESTRICT output_data, const TensorIdx input_stride,
    const TensorIdx output_stride, DeducedRegType<FloatType> &reg_alpha,
    DeducedRegType<FloatType> &reg_beta);


/*
 * Import implementation
 */
#include "kernel_trans.tcc"

}

#endif // HPTC_KERNELS_KERNEL_TRANS_H_
