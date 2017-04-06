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

/*
 * Definition of register's size
 */
constexpr TensorUInt SIZE_REG = 32;


/*
 * Floating types + kernel types selector
 */
template <typename FloatType,
          KernelTypeTrans TYPE>
struct TypeSelector {
  static constexpr bool fl_sc = (std::is_same<float, FloatType>::value or
          std::is_same<FloatComplex, FloatType>::value) and
      (TYPE == KernelTypeTrans::KERNEL_FULL or
          TYPE == KernelTypeTrans::KERNEL_LINE);

  static constexpr bool fl_dz = (std::is_same<double, FloatType>::value or
          std::is_same<DoubleComplex, FloatType>::value) and
      (TYPE == KernelTypeTrans::KERNEL_FULL or
          TYPE == KernelTypeTrans::KERNEL_LINE);

  static constexpr bool h_sc = (std::is_same<float, FloatType>::value or
          std::is_same<FloatComplex, FloatType>::value) and
      TYPE == KernelTypeTrans::KERNEL_HALF;

  static constexpr bool h_d = std::is_same<double, FloatType>::value and
      TYPE == KernelTypeTrans::KERNEL_HALF;

  static constexpr bool h_z = std::is_same<DoubleComplex, FloatType>::value
      and TYPE == KernelTypeTrans::KERNEL_HALF;
};


/*
 * Register types deducer
 */
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


/*
 * Intrinsics wrappers
 */
template <typename FloatType,
          KernelTypeTrans TYPE,
          typename Selected = void>
struct IntrinImpl {
};

template <typename FloatType,
          KernelTypeTrans TYPE>
struct IntrinImpl<FloatType, TYPE,
    Enable<TypeSelector<FloatType, TYPE>::fl_sc>> {
  using Deduced = DeducedFloatType<FloatType>;
  using Reg = RegType<FloatType, TYPE>;

  static HPTC_INL Reg set_reg(const Deduced coef) {
    return _mm256_set1_ps(coef);
  }
  static HPTC_INL Reg load(const FloatType * RESTRICT target) {
    return _mm256_loadu_ps(reinterpret_cast<const Deduced *>(target));
  }
  static HPTC_INL void store(FloatType * RESTRICT target, const Reg &reg) {
    _mm256_storeu_ps(reinterpret_cast<Deduced *>(target), reg);
  }
  static HPTC_INL Reg add(const Reg &reg_a, const Reg &reg_b) {
    return _mm256_add_ps(reg_a, reg_b);
  }
  static HPTC_INL Reg mul(const Reg &reg_a, const Reg &reg_b) {
    return _mm256_mul_ps(reg_a, reg_b);
  }
};

template <typename FloatType,
          KernelTypeTrans TYPE>
struct IntrinImpl<FloatType, TYPE,
    Enable<TypeSelector<FloatType, TYPE>::fl_dz>> {
  using Deduced = DeducedFloatType<FloatType>;
  using Reg = RegType<FloatType, TYPE>;

  static HPTC_INL Reg set_reg(const Deduced coef) {
    return _mm256_set1_pd(coef);
  }
  static HPTC_INL Reg load(const FloatType * RESTRICT target) {
    return _mm256_loadu_pd(reinterpret_cast<const Deduced *>(target));
  }
  static HPTC_INL void store(FloatType * RESTRICT target, const Reg &reg) {
    _mm256_storeu_pd(reinterpret_cast<Deduced *>(target), reg);
  }
  static HPTC_INL Reg add(const Reg &reg_a, const Reg &reg_b) {
    return _mm256_add_pd(reg_a, reg_b);
  }
  static HPTC_INL Reg mul(const Reg &reg_a, const Reg &reg_b) {
    return _mm256_mul_pd(reg_a, reg_b);
  }
};

template <typename FloatType,
          KernelTypeTrans TYPE>
struct IntrinImpl<FloatType, TYPE,
    Enable<TypeSelector<FloatType, TYPE>::h_sc>> {
  using Deduced = DeducedFloatType<FloatType>;
  using Reg = RegType<FloatType, TYPE>;

  static HPTC_INL RegType<FloatType, TYPE> set_reg(const Deduced coef) {
    return _mm_set1_ps(coef);
  }
  static HPTC_INL Reg load(const FloatType * RESTRICT target) {
    return _mm_loadu_ps(reinterpret_cast<const Deduced *>(target));
  }
  static HPTC_INL void store(FloatType * RESTRICT target, const Reg &reg) {
    _mm_storeu_ps(reinterpret_cast<Deduced *>(target), reg);
  }
  static HPTC_INL Reg add(const Reg &reg_a, const Reg &reg_b) {
    return _mm_add_ps(reg_a, reg_b);
  }
  static HPTC_INL Reg mul(const Reg &reg_a, const Reg &reg_b) {
    return _mm_mul_ps(reg_a, reg_b);
  }
};

template <typename FloatType,
          KernelTypeTrans TYPE>
struct IntrinImpl<FloatType, TYPE, Enable<TypeSelector<FloatType, TYPE>::h_d>> {
  using Deduced = DeducedFloatType<FloatType>;
  using Reg = RegType<FloatType, TYPE>;

  static HPTC_INL RegType<FloatType, TYPE> set_reg(const Deduced coef) {
    return _mm_set1_pd(coef);
  }
  static HPTC_INL Reg load(const FloatType * RESTRICT target) {
    return _mm_loadu_pd(reinterpret_cast<const Deduced *>(target));
  }
  static HPTC_INL void store(FloatType * RESTRICT target, const Reg &reg) {
    _mm_storeu_pd(reinterpret_cast<Deduced *>(target), reg);
  }
  static HPTC_INL Reg add(const Reg &reg_a, const Reg &reg_b) {
    return _mm_add_pd(reg_a, reg_b);
  }
  static HPTC_INL Reg mul(const Reg &reg_a, const Reg &reg_b) {
    return _mm_mul_pd(reg_a, reg_b);
  }
};

template <typename FloatType,
          KernelTypeTrans TYPE>
struct IntrinImpl<FloatType, TYPE, Enable<TypeSelector<FloatType, TYPE>::h_z>> {
  using Deduced = DeducedFloatType<FloatType>;
  using Reg = RegType<FloatType, TYPE>;

  static HPTC_INL RegType<FloatType, TYPE> set_reg(const Deduced coef) {
    return coef;
  }
};


/*
 * Kernel base class for storing kernel data
 */
template <typename FloatType,
          KernelTypeTrans TYPE>
class KernelTransData {
public:
  using Float = FloatType;

  KernelTransData();

  static constexpr TensorUInt KN_WIDTH = TYPE == KernelTypeTrans::KERNEL_FULL
      ? SIZE_REG / sizeof(FloatType) : TYPE == KernelTypeTrans::KERNEL_HALF
      ? (SIZE_REG / sizeof(FloatType)) / 2 : 1;

  void set_coef(const DeducedFloatType<FloatType> alpha,
      const DeducedFloatType<FloatType> beta);

protected:
  RegType<FloatType, TYPE> reg_alpha_, reg_beta_;
  DeducedFloatType<FloatType> alpha_, beta_;
};


/*
 * Transpose kernel class
 */
template <typename FloatType,
          KernelTypeTrans TYPE>
class KernelTrans : public KernelTransData<FloatType, TYPE> {
public:
  using Float = FloatType;

  KernelTrans();

  void exec(const FloatType * RESTRICT data_in,
      FloatType * RESTRICT data_out, const TensorIdx stride_in_outld,
      const TensorIdx stride_out_inld) const;
};


/*
 * Import specializations for class KernelTrans and explicit template
 * instantiation for classes KernelTrans and KernelTransData
 */
#include "kernel_trans_avx2.tcc"

}

#endif // HPTC_ARCH_AVX2_KERNEL_TRANS_AVX2_H_
