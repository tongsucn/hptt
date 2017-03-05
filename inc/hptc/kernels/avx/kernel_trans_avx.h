#pragma once
#ifndef HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_H_
#define HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_H_

#include <xmmintrin.h>
#include <immintrin.h>

#include <type_traits>
#include <algorithm>

#include <hptc/types.h>
#include <hptc/config/config_trans.h>


namespace hptc {

#define REG_SIZE_BYTE_AVX 32


template <typename FloatType,
          KernelTypeTrans TYPE>
struct RegTypeDeducer<FloatType, TYPE,
    std::enable_if_t<(std::is_same<FloatType, float>::value or
            std::is_same<FloatType, FloatComplex>::value) and
        (TYPE == KernelTypeTrans::KERNEL_FULL or
            TYPE == KernelTypeTrans::KERNEL_LINE)>> {
  using type = __m256;
};

template <typename FloatType,
          KernelTypeTrans TYPE>
struct RegTypeDeducer<FloatType, TYPE,
    std::enable_if_t<(std::is_same<FloatType, double>::value or
            std::is_same<FloatType, DoubleComplex>::value) and
        (TYPE == KernelTypeTrans::KERNEL_FULL or
            TYPE == KernelTypeTrans::KERNEL_LINE)>> {
  using type = __m256d;
};

template <typename FloatType,
          KernelTypeTrans TYPE>
struct RegTypeDeducer<FloatType, TYPE,
    std::enable_if_t<(std::is_same<FloatType, float>::value or
            std::is_same<FloatType, FloatComplex>::value) and
        TYPE == KernelTypeTrans::KERNEL_HALF>> {
  using type = __m128;
};

template <typename FloatType,
          KernelTypeTrans TYPE>
struct RegTypeDeducer<FloatType, TYPE,
    std::enable_if_t<std::is_same<FloatType, double>::value and
        TYPE == KernelTypeTrans::KERNEL_HALF>> {
  using type = __m128d;
};

template <typename FloatType,
          KernelTypeTrans TYPE>
struct RegTypeDeducer<FloatType, TYPE,
    std::enable_if_t<std::is_same<FloatType, DoubleComplex>::value and
            TYPE == KernelTypeTrans::KERNEL_HALF>> {
  using type = double;
};


template <typename FloatType,
          KernelTypeTrans TYPE>
struct KernelTransAvxBase {
  using RegType = DeducedRegType<FloatType, TYPE>;
  using Deduced = DeducedFloatType<FloatType>;

  KernelTransAvxBase(Deduced coef_alpha, Deduced coef_beta);

  INLINE GenNumType get_kernel_width();
  INLINE GenNumType get_reg_num();

  template <KernelTypeTrans KERNEL = TYPE,
            std::enable_if_t<KERNEL == KernelTypeTrans::KERNEL_FULL or
                KERNEL == KernelTypeTrans::KERNEL_LINE> * = nullptr>
  INLINE DeducedRegType<float, KERNEL> reg_coef(float coef);

  template <KernelTypeTrans KERNEL = TYPE,
            std::enable_if_t<KERNEL == KernelTypeTrans::KERNEL_FULL or
                KERNEL == KernelTypeTrans::KERNEL_LINE> * = nullptr>
  INLINE DeducedRegType<double, KERNEL> reg_coef(double coef);

  template <KernelTypeTrans KERNEL = TYPE,
            std::enable_if_t<KERNEL == KernelTypeTrans::KERNEL_HALF> *
                = nullptr>
  INLINE DeducedRegType<float, KERNEL> reg_coef(float coef);

  template <KernelTypeTrans KERNEL = TYPE,
            std::enable_if_t<std::is_same<FloatType, double>::value and
                KERNEL == KernelTypeTrans::KERNEL_HALF> * = nullptr>
  INLINE DeducedRegType<double, KERNEL> reg_coef(double coef);

  template <KernelTypeTrans KERNEL = TYPE,
            std::enable_if_t<std::is_same<FloatType, DoubleComplex>::value and
                KERNEL == KernelTypeTrans::KERNEL_HALF> * = nullptr>
  INLINE DeducedRegType<DoubleComplex, KERNEL> reg_coef(double coef);

  RegType reg_alpha, reg_beta;
};


template <typename FloatType,
          CoefUsageTrans USAGE,
          KernelTypeTrans TYPE>
struct KernelTransAvx final : public KernelTransAvxBase<FloatType, TYPE> {
};


/*
 * Import implementation
 */
#include "kernel_trans_avx.tcc"

}

#endif // HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_H_
