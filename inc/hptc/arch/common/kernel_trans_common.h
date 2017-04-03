#pragma once
#ifndef HPTC_ARCH_COMMON_KERNEL_TRANS_COMMON_H_
#define HPTC_ARCH_COMMON_KERNEL_TRANS_COMMON_H_

#include <hptc/types.h>
#include <hptc/arch/compat.h>
#include <hptc/util/util_trans.h>


namespace hptc {

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
  using Float = FloatType;

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
          KernelTypeTrans TYPE>
class KernelTrans : public KernelTransData<FloatType, TYPE> {
public:
  using Float = FloatType;

  void exec(const FloatType * RESTRICT data_in, FloatType * RESTRICT data_out,
      const TensorIdx stride_in_outld, const TensorIdx stride_out_inld) const;
};


/*
 * Specialization of class KernelTrans
 */
template <typename FloatType>
class KernelTrans<FloatType, KernelTypeTrans::KERNEL_LINE>
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
 * Explicit template instantiation declaration for class KernelTransData
 */
extern template class KernelTransData<float, KernelTypeTrans::KERNEL_FULL>;
extern template class KernelTransData<double, KernelTypeTrans::KERNEL_FULL>;
extern template class KernelTransData<FloatComplex,
    KernelTypeTrans::KERNEL_FULL>;
extern template class KernelTransData<DoubleComplex,
    KernelTypeTrans::KERNEL_FULL>;

extern template class KernelTransData<float, KernelTypeTrans::KERNEL_HALF>;
extern template class KernelTransData<double, KernelTypeTrans::KERNEL_HALF>;
extern template class KernelTransData<FloatComplex,
    KernelTypeTrans::KERNEL_HALF>;
extern template class KernelTransData<DoubleComplex,
    KernelTypeTrans::KERNEL_HALF>;

extern template class KernelTransData<float, KernelTypeTrans::KERNEL_LINE>;
extern template class KernelTransData<double, KernelTypeTrans::KERNEL_LINE>;
extern template class KernelTransData<FloatComplex,
    KernelTypeTrans::KERNEL_LINE>;
extern template class KernelTransData<DoubleComplex,
    KernelTypeTrans::KERNEL_LINE>;


/*
 * Explicit template instantiation declaration for class KernelTransData and
 * KernelTrans
 */
extern template class KernelTrans<float, KernelTypeTrans::KERNEL_FULL>;
extern template class KernelTrans<double, KernelTypeTrans::KERNEL_FULL>;
extern template class KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_FULL>;
extern template class KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_FULL>;

extern template class KernelTrans<float, KernelTypeTrans::KERNEL_HALF>;
extern template class KernelTrans<double, KernelTypeTrans::KERNEL_HALF>;
extern template class KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_HALF>;
extern template class KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_HALF>;

extern template class KernelTrans<float, KernelTypeTrans::KERNEL_LINE>;
extern template class KernelTrans<double, KernelTypeTrans::KERNEL_LINE>;
extern template class KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_LINE>;
extern template class KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_LINE>;


}

#endif // HPTC_ARCH_COMMON_KERNEL_TRANS_COMMON_H_
