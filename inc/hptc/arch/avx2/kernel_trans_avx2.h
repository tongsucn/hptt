#pragma once
#ifndef HPTC_ARCH_AVX2_KERNEL_TRANS_AVX2_H_
#define HPTC_ARCH_AVX2_KERNEL_TRANS_AVX2_H_

#include <type_traits>

#include <immintrin.h>
#include <xmmintrin.h>

#include <hptc/types.h>
#include <hptc/arch/compat.h>
#include <hptc/util/util.h>
#include <hptc/util/util_trans.h>


namespace hptc {

constexpr TensorUInt REG_SIZE = 32;


template <typename FloatType,
          KernelTypeTrans IN_TYPE>
struct TypeSelector {
  static constexpr bool fl_sc = (std::is_same<float, FloatType>::value or
          std::is_same<FloatComplex, FloatType>::value) and
      (IN_TYPE == KernelTypeTrans::KERNEL_FULL or
          IN_TYPE == KernelTypeTrans::KERNEL_LINE);

  static constexpr bool fl_dz = (std::is_same<double, FloatType>::value or
          std::is_same<DoubleComplex, FloatType>::value) and
      (IN_TYPE == KernelTypeTrans::KERNEL_FULL or
          IN_TYPE == KernelTypeTrans::KERNEL_LINE);

  static constexpr bool h_sc = (std::is_same<float, FloatType>::value or
          std::is_same<FloatComplex, FloatType>::value) and
      IN_TYPE == KernelTypeTrans::KERNEL_HALF;

  static constexpr bool h_d = std::is_same<double, FloatType>::value and
      IN_TYPE == KernelTypeTrans::KERNEL_HALF;

  static constexpr bool h_z = std::is_same<DoubleComplex, FloatType>::value
      and IN_TYPE == KernelTypeTrans::KERNEL_HALF;
};


template <typename FloatType,
          KernelTypeTrans TYPE,
          typename Selected = void>
struct RegDeducer {
};

template <typename FloatType,
          KernelTypeTrans TYPE>
struct RegDeducer<FloatType, TYPE,
    Enable<TypeSelector<FloatType, TYPE>::fl_sc>> {
  using type = __m256;
};

template <typename FloatType,
          KernelTypeTrans TYPE>
struct RegDeducer<FloatType, TYPE,
    Enable<TypeSelector<FloatType, TYPE>::fl_dz>> {
  using type = __m256d;
};

template <typename FloatType,
          KernelTypeTrans TYPE>
struct RegDeducer<FloatType, TYPE,
    Enable<TypeSelector<FloatType, TYPE>::h_sc>> {
  using type = __m128;
};

template <typename FloatType,
          KernelTypeTrans TYPE>
struct RegDeducer<FloatType, TYPE,
    Enable<TypeSelector<FloatType, TYPE>::h_d>> {
  using type = __m128d;
};

template <typename FloatType,
          KernelTypeTrans TYPE>
struct RegDeducer<FloatType, TYPE,
    Enable<TypeSelector<FloatType, TYPE>::h_z>> {
  using type = double;
};

template <typename FloatType,
          KernelTypeTrans TYPE>
using RegType = typename RegDeducer<FloatType, TYPE>::type;


template <typename FloatType,
          KernelTypeTrans TYPE>
class KernelTransData {
public:
  using Float = FloatType;

  static constexpr TensorUInt KN_WIDTH = TYPE == KernelTypeTrans::KERNEL_FULL
      ? REG_SIZE / sizeof(FloatType) : TYPE == KernelTypeTrans::KERNEL_HALF
      ? (REG_SIZE / sizeof(FloatType)) / 2 : 1;

  void set_coef(const DeducedFloatType<FloatType> alpha,
      const DeducedFloatType<FloatType> beta);

protected:
  RegType<FloatType, TYPE> reg_alpha_, reg_beta_;
  DeducedFloatType<FloatType> alpha_, beta_;
};


template <typename FloatType,
          KernelTypeTrans TYPE>
class KernelTrans : public KernelTransData<FloatType, TYPE> {
public:
  using Float = FloatType;

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

#endif // HPTC_ARCH_AVX2_KERNEL_TRANS_AVX2_H_