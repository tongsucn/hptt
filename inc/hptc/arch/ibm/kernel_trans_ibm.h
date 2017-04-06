#pragma once
#ifndef HPTC_ARCH_IBM_KERNEL_TRANS_IBM_H_
#define HPTC_ARCH_IBM_KERNEL_TRANS_IBM_H_

#include <type_traits>

#include <builtins.h>
#include <altivec.h>

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
  static constexpr bool fhl_sc = std::is_same<float, FloatType>::value or
          std::is_same<FloatComplex, FloatType>::value;

  static constexpr bool f_d = TYPE == KernelTypeTrans::KERNEL_FULL and
      std::is_same<double, FloatType>::value;

  static constexpr bool fhl_dz = ((std::is_same<double, FloatType>::value or
          std::is_same<DoubleComplex, FloatType>::value) and
      (TYPE == KernelTypeTrans::KERNEL_HALF or
          TYPE == KernelTypeTrans::KERNEL_LINE)) or
      (std::is_same<double, FloatType>::value and
          TYPE == KernelTypeTrans::KERNEL_FULL);
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
    Enable<TypeSelector<FloatType, TYPE>::fhl_sc>> {
  using type = float;
};

template <typename FloatType,
          KernelTypeTrans TYPE>
struct RegDeducer<FloatType, TYPE,
    Enable<TypeSelector<FloatType, TYPE>::f_d>> {
  using type = vector4double;
};

template <typename FloatType,
          KernelTypeTrans TYPE>
struct RegDeducer<FloatType, TYPE,
    Enable<TypeSelector<FloatType, TYPE>::fhl_dz>> {
  using type = double;
};

template <typename FloatType,
          KernelTypeTrans TYPE>
using RegType = typename RegDeducer<FloatType, TYPE>::type;


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
#include "kernel_trans_ibm.tcc"

}

#endif // HPTC_ARCH_IBM_KERNEL_TRANS_IBM_H_
