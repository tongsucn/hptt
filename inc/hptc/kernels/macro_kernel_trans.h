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
  using Float = typename KernelFunc::Float;
  using RegType = typename KernelFunc::RegType;

  void set_coef(const DeducedFloatType<Float> alpha,
      const DeducedFloatType<Float> beta);

  TensorUInt get_cont_len() const;
  TensorUInt get_ncont_len() const;

  void exec(const Float * RESTRICT input_data, Float * RESTRICT output_data,
      const TensorIdx input_stride, const TensorIdx output_stride) const;

private:
  template <TensorUInt CONT,
            TensorUInt NCONT>
  void ncont_tiler_(DualCounter<CONT, NCONT>,
      const Float * RESTRICT input_data, Float * RESTRICT output_data,
      const TensorIdx input_stride, const TensorIdx output_stride) const;

  template <TensorUInt CONT>
  void ncont_tiler_(DualCounter<CONT, 0>,
      const Float * RESTRICT input_data, Float * RESTRICT output_data,
      const TensorIdx input_stride, const TensorIdx output_stride) const;

  template <TensorUInt CONT,
            TensorUInt NCONT>
  void cont_tiler_(DualCounter<CONT, NCONT>,
      const Float * RESTRICT input_data, Float * RESTRICT output_data,
      const TensorIdx input_stride, const TensorIdx output_stride) const;

  template <TensorUInt NCONT>
  void cont_tiler_(DualCounter<0, NCONT>,
      const Float * RESTRICT input_data, Float * RESTRICT output_data,
      const TensorIdx input_stride, const TensorIdx output_stride) const;

  RegType reg_alpha_, reg_beta_;
};


template <typename FloatType,
          CoefUsageTrans USAGE>
class MacroTransLinear {
public:
  using RegType = DeducedRegType<FloatType, KernelTypeTrans::KERNEL_LINE>;

  void set_coef(const DeducedFloatType<FloatType> alpha,
      const DeducedFloatType<FloatType> beta);

  void exec(const FloatType * RESTRICT input_data,
      FloatType * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride) const;

private:
  RegType reg_alpha_, reg_beta_;
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
