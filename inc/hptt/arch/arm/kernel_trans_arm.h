#pragma once
#ifndef HPTT_ARCH_ARM_KERNEL_TRANS_ARM_H_
#define HPTT_ARCH_ARM_KERNEL_TRANS_ARM_H_

#include <hptt/types.h>
#include <hptt/arch/compat.h>
#include <hptt/util/util_trans.h>


namespace hptt {

/*
 * Definition of register's size
 */
constexpr TensorUInt SIZE_REG = 32;


/*
 * Kernel base class for storing kernel data
 */
template <typename FloatType,
          KernelTypeTrans TYPE>
class KernelTransData {
public:
  KernelTransData();

  static constexpr TensorUInt KN_WIDTH = TYPE == KernelTypeTrans::KERNEL_FULL
      ? SIZE_REG / sizeof(FloatType) : TYPE == KernelTypeTrans::KERNEL_HALF
      ? (SIZE_REG / sizeof(FloatType)) / 2 : 1;

  static void sstore(FloatType *data_out, const FloatType *buffer);
  static bool check_stream(TensorUInt);

  void set_coef(const DeducedFloatType<FloatType> alpha,
      const DeducedFloatType<FloatType> beta);

protected:
  DeducedFloatType<FloatType> alpha_, beta_;
};


/*
 * Transpose kernel class
 */
template <typename FloatType,
          KernelTypeTrans TYPE,
          bool UPDATE_OUT>
class KernelTrans : public KernelTransData<FloatType, TYPE> {
public:
  using Float = FloatType;
  static constexpr bool UPDATE = UPDATE_OUT;

  KernelTrans();

  void exec(const FloatType * RESTRICT data_in, FloatType * RESTRICT data_out,
      const TensorIdx stride_in_outld, const TensorIdx stride_out_inld) const;
};


/*
 * Specialization of class KernelTrans
 */
template <typename FloatType,
          bool UPDATE_OUT>
class KernelTrans<FloatType, KernelTypeTrans::KERNEL_LINE, UPDATE_OUT>
    : public KernelTransData<FloatType, KernelTypeTrans::KERNEL_LINE> {
public:
  KernelTrans();

  void exec(const FloatType * RESTRICT data_in, FloatType * RESTRICT data_out,
      const TensorIdx size_trans, const TensorIdx size_pad) const;
};


/*
 * Implementation of class KernelTransData
 */
template <typename FloatType,
          KernelTypeTrans TYPE>
KernelTransData<FloatType, TYPE>::KernelTransData() : alpha_(), beta_() {
}


template <typename FloatType,
          KernelTypeTrans TYPE>
void KernelTransData<FloatType, TYPE>::sstore(FloatType *data_out,
    const FloatType *buffer) {
}


template <typename FloatType,
          KernelTypeTrans TYPE>
bool KernelTransData<FloatType, TYPE>::check_stream(TensorUInt) {
  return false;
}


template <typename FloatType,
          KernelTypeTrans TYPE>
void KernelTransData<FloatType, TYPE>::set_coef(
    const DeducedFloatType<FloatType> alpha,
    const DeducedFloatType<FloatType> beta) {
  this->alpha_ = alpha, this->beta_ = beta;
}


/*
 * Explicit template instantiation declaration for class and KernelTrans
 */
extern template class KernelTrans<float, KernelTypeTrans::KERNEL_FULL, true>;
extern template class KernelTrans<double, KernelTypeTrans::KERNEL_FULL, true>;
extern template class KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_FULL,
    true>;
extern template class KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_FULL,
    true>;

extern template class KernelTrans<float, KernelTypeTrans::KERNEL_HALF, true>;
extern template class KernelTrans<double, KernelTypeTrans::KERNEL_HALF, true>;
extern template class KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_HALF,
    true>;
extern template class KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_HALF,
    true>;

extern template class KernelTrans<float, KernelTypeTrans::KERNEL_LINE, true>;
extern template class KernelTrans<double, KernelTypeTrans::KERNEL_LINE, true>;
extern template class KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_LINE,
    true>;
extern template class KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_LINE,
    true>;

extern template class KernelTrans<float, KernelTypeTrans::KERNEL_FULL, false>;
extern template class KernelTrans<double, KernelTypeTrans::KERNEL_FULL, false>;
extern template class KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_FULL,
    false>;
extern template class KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_FULL,
    false>;

extern template class KernelTrans<float, KernelTypeTrans::KERNEL_HALF, false>;
extern template class KernelTrans<double, KernelTypeTrans::KERNEL_HALF, false>;
extern template class KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_HALF,
    false>;
extern template class KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_HALF,
    false>;

extern template class KernelTrans<float, KernelTypeTrans::KERNEL_LINE, false>;
extern template class KernelTrans<double, KernelTypeTrans::KERNEL_LINE, false>;
extern template class KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_LINE,
    false>;
extern template class KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_LINE,
    false>;

}

#endif // HPTT_ARCH_ARM_KERNEL_TRANS_ARM_H_
