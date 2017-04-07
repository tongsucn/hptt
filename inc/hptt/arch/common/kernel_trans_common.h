#pragma once
#ifndef HPTT_ARCH_COMMON_KERNEL_TRANS_COMMON_H_
#define HPTT_ARCH_COMMON_KERNEL_TRANS_COMMON_H_

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
  static constexpr bool STREAM = false;

  KernelTransData();

  static constexpr TensorUInt KN_WIDTH = TYPE == KernelTypeTrans::KERNEL_FULL
      ? SIZE_REG / sizeof(FloatType) : TYPE == KernelTypeTrans::KERNEL_HALF
      ? (SIZE_REG / sizeof(FloatType)) / 2 : 1;

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
  static constexpr TensorUInt LOOP_MAX = 10;

  KernelTrans();

  void set_wrapper_loop(const TensorIdx stride_in_inld,
      const TensorIdx stride_in_outld, const TensorIdx stride_out_inld,
      const TensorIdx stride_out_outld, const TensorUInt size_kn_inld,
      const TensorUInt size_kn_outld);

  void exec(const FloatType * RESTRICT data_in, FloatType * RESTRICT data_out,
      const TensorIdx size_trans, const TensorIdx size_pad) const;

private:
  TensorIdx stride_in_inld_, stride_in_outld_, stride_out_inld_,
      stride_out_outld_;
  TensorUInt size_kn_inld_, size_kn_outld_;
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

#endif // HPTT_ARCH_COMMON_KERNEL_TRANS_COMMON_H_
