#pragma once
#ifndef HPTC_ARCH_COMMON_KERNEL_TRANS_COMMON_H_
#define HPTC_ARCH_COMMON_KERNEL_TRANS_COMMON_H_

#include <hptc/types.h>
#include <hptc/arch/compat.h>
#include <hptc/util/util_trans.h>


namespace hptc {

constexpr TensorUInt REG_SIZE = 32;


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
  DeducedFloatType<FloatType> alpha_, beta_;
};


template <typename FloatType,
          KernelTypeTrans TYPE>
class KernelTrans : public KernelTransData<FloatType, TYPE> {
public:
  using Float = FloatType;

  void exec(const FloatType * RESTRICT in_data, FloatType * RESTRICT out_data,
      const TensorIdx input_stride, const TensorIdx output_stride) const;
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

  void set_wrapper_loop(const TensorIdx stride_in_in,
      const TensorIdx stride_in_out, const TensorIdx stride_out_in,
      const TensorIdx stride_out_out, const TensorUInt ld_in_size,
      const TensorUInt ld_out_size);

  void exec(const FloatType * RESTRICT in_data, FloatType * RESTRICT out_data,
      const TensorIdx in_size, const TensorIdx out_size) const;

private:
  TensorIdx stride_in_in_, stride_in_out_, stride_out_in_, stride_out_out_;
  TensorUInt ld_in_size_, ld_out_size_;
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
