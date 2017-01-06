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


/*
 * Import implementation
 */
#include "kernel_trans_avx.tcc"

}

#endif // HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_H_
