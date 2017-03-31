#pragma once
#ifndef HPTC_ARCH_AVX2_KERNEL_TRANS_AVX_H_
#define HPTC_ARCH_AVX2_KERNEL_TRANS_AVX_H_

#include <immintrin.h>
#include <xmmintrin.h>

#include <hptc/types.h>
#include <hptc/arch/compat.h>
#include <hptc/util/util_trans.h>


namespace hptc {

template <typename FloatType,
          KernelTypeTrans TYPE>
struct DeducedRegType {
};

template <>
struct DeducedRegType<float, KernelTypeTrans::KERNEL_FULL> {
  using type = __m256;
};

template <>
struct DeducedRegType<double, KernelTypeTrans::KERNEL_FULL> {
  using type = __m256d;
};

template <>
struct DeducedRegType<FloatComplex, KernelTypeTrans::KERNEL_FULL> {
  using type = __m256;
};

template <>
struct DeducedRegType<DoubleComplex, KernelTypeTrans::KERNEL_FULL> {
  using type = __m256d;
};

template <>
struct DeducedRegType<float, KernelTypeTrans::KERNEL_HALF> {
  using type = __m128;
};

template <>
struct DeducedRegType<double, KernelTypeTrans::KERNEL_HALF> {
  using type = __m128d;
};

template <>
struct DeducedRegType<FloatComplex, KernelTypeTrans::KERNEL_HALF> {
  using type = __m128;
};

template <>
struct DeducedRegType<DoubleComplex, KernelTypeTrans::KERNEL_HALF> {
  using type = double;
};

template <>
struct DeducedRegType<float, KernelTypeTrans::KERNEL_LINE> {
  using type = float;
};

template <>
struct DeducedRegType<double, KernelTypeTrans::KERNEL_LINE> {
  using type = double;
};

template <>
struct DeducedRegType<FloatComplex, KernelTypeTrans::KERNEL_LINE> {
  using type = float;
};

template <>
struct DeducedRegType<DoubleComplex, KernelTypeTrans::KERNEL_LINE> {
  using type = double;
};

template <typename FloatType,
          KernelTypeTrans TYPE>
using RegType = typename DeducedRegType<FloatType, TYPE>::type;



template <typename FloatType,
          KernelTypeTrans TYPE>
class KernelTransData {
public:
  using Float = FloatType;

  static constexpr TensorUInt KN_WIDTH = TYPE == KernelTypeTrans::KERNEL_FULL
      ? 32 / sizeof(FloatType) : TYPE == KernelTypeTrans::KERNEL_HALF
      ? 16 / sizeof(FloatType) : 1;

protected:
  RegType<FloatType, TYPE> reg_alpha_, reg_beta_;
};


template <typename FloatType,
          KernelTypeTrans TYPE>
class KernelTrans : public KernelTransData<FloatType, TYPE> {
public:
  using Float = FloatType;

  void set_coef(const DeducedFloatType<FloatType> alpha,
      const DeducedFloatType<FloatType> beta);

  void exec(const FloatType * RESTRICT input_data,
      FloatType * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride) const;
};


/*
 * Import specializations for class KernelTrans and explicit template
 * instantiation for class KernelTrans and KernelTransData
 */
#include "kernel_trans_avx2.tcc"

}

#endif // HPTC_ARCH_AVX2_KERNEL_TRANS_AVX_H_
