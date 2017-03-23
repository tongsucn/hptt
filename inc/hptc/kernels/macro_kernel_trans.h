#pragma once
#ifndef HPTC_KERNELS_MACRO_KERNEL_TRANS_H_
#define HPTC_KERNELS_MACRO_KERNEL_TRANS_H_

#include <hptc/types.h>
#include <hptc/util/util_trans.h>
#include <hptc/kernels/micro_kernel_trans.h>


namespace hptc {

template <typename KernelFunc,
          TensorUInt CONT_LEN,
          TensorUInt NCONT_LEN>
class MacroTransVec {
public:
  using FloatType = typename KernelFunc::FloatType;
  using RegType = typename KernelFunc::RegType;

  static RegType reg_coef(const DeducedFloatType<FloatType> coef);
  constexpr TensorUInt get_cont_len() const;
  constexpr TensorUInt get_ncont_len() const;

  void operator()(const FloatType * RESTRICT input_data,
      FloatType * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride, const RegType &reg_alpha,
      const RegType &reg_beta) const;

private:
  KernelFunc kernel_;
};


template <typename FloatType,
          CoefUsageTrans USAGE>
class MacroTransLinear {
public:
  using RegType = DeducedRegType<FloatType, KernelTypeTrans::KERNEL_LINE>;

  static RegType reg_coef(const DeducedFloatType<FloatType> coef);
  void operator()(const FloatType * RESTRICT input_data,
      FloatType * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride, const RegType &alpha,
      const RegType &beta) const;
};


/*
 * Alias of macro kernels
 */
template <typename FloatType,
          CoefUsageTrans USAGE,
          TensorUInt CONT_LEN,
          TensorUInt NCONT_LEN>
using MacroTransVecFull = MacroTransVec<KernelTransFull<FloatType, USAGE>,
      CONT_LEN, NCONT_LEN>;


template <typename FloatType,
          CoefUsageTrans USAGE,
          TensorUInt CONT_LEN,
          TensorUInt NCONT_LEN>
using MacroTransVecHalf = MacroTransVec<KernelTransHalf<FloatType, USAGE>,
      CONT_LEN, NCONT_LEN>;


/*
 * Import explicit instantiation declaration
 */
#include "macro_kernel_trans.tcc"

}

#endif // HPTC_KERNELS_MACRO_KERNEL_TRANS_H_
