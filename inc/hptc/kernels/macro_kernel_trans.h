#pragma once
#ifndef HPTC_KERNELS_MACRO_KERNEL_TRANS_H_
#define HPTC_KERNELS_MACRO_KERNEL_TRANS_H_

#include <hptc/util.h>
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

  template <GenNumType CONT,
            GenNumType NCONT>
  INLINE void ncont_tiler(DualCounter<CONT, NCONT>,
      const FloatType * RESTRICT input_data, FloatType * RESTRICT output_data,
      const TensorIdx input_stride, const TensorIdx output_stride);

  template <GenNumType CONT>
  INLINE void ncont_tiler(DualCounter<CONT, 0>,
      const FloatType * RESTRICT input_data, FloatType * RESTRICT output_data,
      const TensorIdx input_stride, const TensorIdx output_stride);

  template <GenNumType CONT,
            GenNumType NCONT>
  INLINE void cont_tiler(DualCounter<CONT, NCONT>,
      const FloatType * RESTRICT input_data, FloatType * RESTRICT output_data,
      const TensorIdx input_stride, const TensorIdx output_stride);

  template <GenNumType NCONT>
  INLINE void cont_tiler(DualCounter<0, NCONT>,
      const FloatType * RESTRICT input_data, FloatType * RESTRICT output_data,
      const TensorIdx input_stride, const TensorIdx output_stride);


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
      DeducedFloatType<FloatType> beta);

  INLINE void operator()(const FloatType * RESTRICT input_data,
      FloatType * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride);
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
