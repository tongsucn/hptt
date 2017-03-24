#pragma once
#ifndef HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_H_
#define HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_H_

#include <xmmintrin.h>
#include <immintrin.h>

#include <type_traits>

#include <hptc/types.h>
#include <hptc/util/util.h>
#include <hptc/util/util_trans.h>


namespace hptc {

#define REG_SIZE_BYTE_AVX 32


template <typename FloatType,
          KernelTypeTrans TYPE>
struct RegTypeDeducer<FloatType, TYPE,
    hptc::Enable<(std::is_same<FloatType, float>::value or
            std::is_same<FloatType, FloatComplex>::value) and
        TYPE == KernelTypeTrans::KERNEL_FULL>> {
  using type = __m256;
};

template <typename FloatType,
          KernelTypeTrans TYPE>
struct RegTypeDeducer<FloatType, TYPE,
    hptc::Enable<(std::is_same<FloatType, double>::value or
            std::is_same<FloatType, DoubleComplex>::value) and
        TYPE == KernelTypeTrans::KERNEL_FULL>> {
  using type = __m256d;
};

template <typename FloatType,
          KernelTypeTrans TYPE>
struct RegTypeDeducer<FloatType, TYPE,
    hptc::Enable<(std::is_same<FloatType, float>::value or
            std::is_same<FloatType, FloatComplex>::value) and
        TYPE == KernelTypeTrans::KERNEL_HALF>> {
  using type = __m128;
};

template <typename FloatType,
          KernelTypeTrans TYPE>
struct RegTypeDeducer<FloatType, TYPE,
    hptc::Enable<std::is_same<FloatType, double>::value and
        TYPE == KernelTypeTrans::KERNEL_HALF>> {
  using type = __m128d;
};

template <typename FloatType,
          KernelTypeTrans TYPE>
struct RegTypeDeducer<FloatType, TYPE,
    hptc::Enable<TYPE == KernelTypeTrans::KERNEL_LINE and
        (std::is_same<FloatType, float>::value or
            std::is_same<FloatType, FloatComplex>::value)>> {
  using type = float;
};

template <typename FloatType,
          KernelTypeTrans TYPE>
struct RegTypeDeducer<FloatType, TYPE,
    hptc::Enable<(TYPE == KernelTypeTrans::KERNEL_LINE and
        (std::is_same<FloatType, double>::value or
            std::is_same<FloatType, DoubleComplex>::value)) or
        (TYPE == KernelTypeTrans::KERNEL_HALF and
            std::is_same<FloatType, DoubleComplex>::value)>> {
  using type = double;
};


template <typename FloatType,
          KernelTypeTrans TYPE>
struct KernelTransAvxBase {
  using FloatType = FloatType;
  using RegType = DeducedRegType<FloatType, TYPE>;

  static constexpr TensorUInt kn_width = KernelTypeTrans::KERNEL_FULL == TYPE
      ? REG_SIZE_BYTE_AVX / sizeof(FloatType)
      : KernelTypeTrans::KERNEL_HALF == TYPE
          ? REG_SIZE_BYTE_AVX / sizeof(FloatType) / 2 : 1;
};


template <typename FloatType,
          CoefUsageTrans USAGE,
          KernelTypeTrans TYPE>
struct KernelTransAvx final : public KernelTransAvxBase<FloatType, TYPE> {
  using RegType = DeducedRegType<FloatType, TYPE>;

  static RegType reg_coef(const DeducedFloatType<FloatType> coef);

  void operator()(const FloatType * RESTRICT input_data,
      FloatType * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride, const RegType &reg_alpha,
      const RegType &reg_beta) const;
};


/*
 * Type alias for AVX micro kernel
 */
template <typename FloatType,
          CoefUsageTrans USAGE,
          KernelTypeTrans TYPE>
using KernelTrans = KernelTransAvx<FloatType, USAGE, TYPE>;



/*
 * Import template class KernelTransAvx's partial specialization
 * and explicit instantiation declaration
 */
#include "kernel_trans_avx.tcc"

}

#endif // HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_H_
