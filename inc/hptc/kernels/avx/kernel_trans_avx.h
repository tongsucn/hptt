#pragma once
#ifndef HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_H_
#define HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_H_

#include <immintrin.h>

#include <type_traits>

#include <hptc/types.h>
#include <hptc/param/parameter_trans.h>
#include <hptc/kernels/kernel_trans_base.h>


namespace hptc {

/*
 * Register type deducing utilities
 */
template <typename FloatType>
struct RegDeducerAvx {
};

template <>
struct RegDeducerAvx<float> {
  using type = __m256;
};

template <>
struct RegDeducerAvx<double> {
  using type = __m256d;
};

template <>
struct RegDeducerAvx<FloatComplex> {
  using type = __m256;
};

template <>
struct RegDeducerAvx<DoubleComplex> {
  using type = __m256d;
};


template <typename FloatType,
          KernelType TYPE>
struct KernelTransAvxBase {
  INLINE GenNumType get_reg_num();
  INLINE __m256 reg_coef(float coef);
  INLINE __m256d reg_coef(double coef);
};


template <typename FloatType,
          CoefUsage USAGE,
          KernelType TYPE = KernelType::KERNEL_FULL>
struct KernelTransAvx final : public KernelTransAvxBase<FloatType, TYPE> {
};


/*
 * Import implementation
 */
#include "kernel_trans_avx.tcc"

}

#endif // HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_H_
