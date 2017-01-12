#pragma once
#ifndef HPTC_KERNELS_MACRO_KERNEL_TRANS_H_
#define HPTC_KERNELS_MACRO_KERNEL_TRANS_H_

#include <hptc/types.h>
#include <hptc/param/parameter_trans.h>
#include <hptc/kernels/kernel_trans.h>


namespace hptc {

template <typename FloatType,
          typename KernelFunc,
          GenNumType CONT_LEN,
          GenNumType NCONT_LEN>
class MacroTransVecData {
public:
  MacroTransVecData(KernelFunc kernel, DeducedFloatType<FloatType> alpha,
      DeducedFloatType<FloatType> beta);

  INLINE GenNumType get_cont_len();
  INLINE GenNumType get_ncont_len();

protected:
  using RegType = typename KernelFunc::RegType;

  KernelFunc kernel_;
  RegType reg_alpha_, reg_beta_;
  GenNumType reg_num_;
};


template <typename FloatType,
          typename KernelFunc,
          GenNumType CONT_LEN,
          GenNumType NCONT_LEN>
class MacroTransVec
    : public MacroTransVecData<FloatType, KernelFunc, CONT_LEN, NCONT_LEN> {
public:
  MacroTransVec(KernelFunc kernel, DeducedFloatType<FloatType> alpha,
      DeducedFloatType<FloatType> beta)
      : MacroTransVecData<FloatType, KernelFunc, CONT_LEN, NCONT_LEN>(kernel,
          alpha, beta) {
  }

  INLINE void operator()(const FloatType * RESTRICT input_data,
      FloatType * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride) {
    this->kernel_(input_data, output_data, input_stride, output_stride,
        this->reg_alpha_, this->reg_beta_);
    this->kernel_(input_data + this->reg_num_ * input_stride,
        output_data + this->reg_num_,
        input_stride, output_stride, this->reg_alpha_, this->reg_beta_);
  }
};


template <typename FloatType,
          CoefUsage USAGE>
class MacroTransScalarData {
public:
  MacroTransScalarData(DeducedFloatType<FloatType> alpha,
      DeducedFloatType<FloatType> beta);

protected:
  DeducedFloatType<FloatType> alpha, beta;
};


template <typename FloatType,
          CoefUsage USAGE>
class MacroTransScalar
    : public MacroTransScalarData<FloatType, USAGE> {
};


/*
 * Import implementation
 */
#include "macro_kernel_trans.tcc"

}

#endif // HPTC_KERNELS_MACRO_KERNEL_TRANS_H_
