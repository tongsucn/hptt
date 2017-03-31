#pragma once
#ifndef HPTC_ARCH_COMMON_KERNEL_TRANS_COMMON_H_
#define HPTC_ARCH_COMMON_KERNEL_TRANS_COMMON_H_

#include <hptc/types.h>
#include <hptc/arch/compat.h>
#include <hptc/util/util_trans.h>


namespace hptc {

template <typename FloatType,
          KernelTypeTrans TYPE>
class KernelTrans {
public:
  using Float = FloatType;

  void set_coef(const DeducedFloatType<FloatType> alpha,
      const DeducedFloatType<FloatType> beta);

  void exec(const FloatType * RESTRICT in_data, FloatType * RESTRICT out_data,
      const TensorIdx input_stride, const TensorIdx output_stride) const;

  static constexpr TensorUInt KN_WIDTH = TYPE == KernelTypeTrans::KERNEL_FULL
      ? 32 / sizeof(FloatType) : TYPE == KernelTypeTrans::KERNEL_HALF
      ? 16 / sizeof(FloatType) : 1;

private:
  DeducedFloatType<FloatType> reg_alpha_, reg_beta_;
};


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
