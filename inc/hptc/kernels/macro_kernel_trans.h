#pragma once
#ifndef HPTC_KERNELS_MACRO_KERNEL_TRANS_H_
#define HPTC_KERNELS_MACRO_KERNEL_TRANS_H_

#include <hptc/util.h>
#include <hptc/types.h>
#include <hptc/config/config_trans.h>
#include <hptc/kernels/kernel_trans.h>


namespace hptc {

template <typename FloatType,
          typename KernelFunc>
class MacroTransVecData {
public:
  MacroTransVecData(DeducedFloatType<FloatType> alpha,
      DeducedFloatType<FloatType> beta);

protected:
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
  GenNumType kn_wd_;    // Kernel width, number of elements in one register
};


template <typename FloatType,
          typename KernelFunc,
          GenNumType CONT_LEN,
          GenNumType NCONT_LEN>
class MacroTransVec : public MacroTransVecData<FloatType, KernelFunc> {
public:
  MacroTransVec(DeducedFloatType<FloatType> alpha,
      DeducedFloatType<FloatType> beta);

  INLINE GenNumType get_cont_len();
  INLINE GenNumType get_ncont_len();

  INLINE void operator()(const FloatType * RESTRICT input_data,
      FloatType * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride);
};


template <typename FloatType,
          CoefUsageTrans USAGE>
class MacroTransScalarData {
public:
  MacroTransScalarData(DeducedFloatType<FloatType> alpha,
      DeducedFloatType<FloatType> beta);

protected:
  DeducedFloatType<FloatType> alpha, beta;
};


template <typename FloatType,
          CoefUsageTrans USAGE>
class MacroTransScalar
    : public MacroTransScalarData<FloatType, USAGE> {
public:
  MacroTransScalar(DeducedFloatType<FloatType> alpha,
      DeducedFloatType<FloatType> beta);
  INLINE void operator()(const FloatType * RESTRICT input_data,
      FloatType * RESTRICT output_data, const TensorIdx input_stride = 0,
      const TensorIdx output_stride = 0);
};


/*
 * Alias and instantiation of vectorized macro kernels
 */
template <typename FloatType,
          CoefUsageTrans USAGE,
          GenNumType CONT_LEN,
          GenNumType NCONT_LEN>
using MacroTransVecFull = MacroTransVec<FloatType,
      KernelTransFull<FloatType, USAGE>, CONT_LEN, NCONT_LEN>;


template <typename FloatType,
          CoefUsageTrans USAGE,
          GenNumType CONT_LEN,
          GenNumType NCONT_LEN>
using MacroTransVecHalf = MacroTransVec<FloatType,
      KernelTransHalf<FloatType, USAGE>, CONT_LEN, NCONT_LEN>;


template <typename FloatType,
          CoefUsageTrans USAGE>
using MacroTransVecFullBig = MacroTransVecFull<FloatType, USAGE, 4, 4>;


template <typename FloatType,
          CoefUsageTrans USAGE>
using MacroTransVecFullVertical = MacroTransVecFull<FloatType, USAGE, 1, 4>;


template <typename FloatType,
          CoefUsageTrans USAGE>
using MacroTransVecFullHorizontal = MacroTransVecFull<FloatType, USAGE, 4, 1>;


template <typename FloatType,
          CoefUsageTrans USAGE>
using MacroTransVecFullSmall = MacroTransVecFull<FloatType, USAGE, 1, 1>;


template <typename FloatType,
          CoefUsageTrans USAGE>
using MacroTransVecHalfVertical = MacroTransVecHalf<FloatType, USAGE, 1, 2>;


template <typename FloatType,
          CoefUsageTrans USAGE>
using MacroTransVecHalfHorizontal = MacroTransVecHalf<FloatType, USAGE, 2, 1>;


template <typename FloatType,
          CoefUsageTrans USAGE>
using MacroTransVecHalfSmall = MacroTransVecHalf<FloatType, USAGE, 1, 1>;


/*
 * Import implementation
 */
#include "macro_kernel_trans.tcc"

}

#endif // HPTC_KERNELS_MACRO_KERNEL_TRANS_H_
