#pragma once
#ifndef HPTC_KERNELS_MACRO_KERNEL_TRANS_H_
#define HPTC_KERNELS_MACRO_KERNEL_TRANS_H_

#include <algorithm>

#include <hptc/util.h>
#include <hptc/types.h>
#include <hptc/config/config_trans.h>
#include <hptc/kernels/kernel_trans.h>


namespace hptc {

template <typename KernelFunc,
          GenNumType CONT_LEN,
          GenNumType NCONT_LEN>
class MacroTransVec {
public:
  using FloatType = typename KernelFunc::FLOAT;
  using Deduced = DeducedFloatType<FloatType>;

  MacroTransVec(Deduced alpha, Deduced beta);

  GenNumType get_cont_len();
  GenNumType get_ncont_len();

  void operator()(const FloatType * RESTRICT input_data,
      FloatType * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride);

private:
  template <GenNumType CONT,
            GenNumType NCONT>
  void ncont_tiler_(DualCounter<CONT, NCONT>,
      const FloatType * RESTRICT input_data, FloatType * RESTRICT output_data,
      const TensorIdx input_stride, const TensorIdx output_stride);

  template <GenNumType CONT>
  void ncont_tiler_(DualCounter<CONT, 0>,
      const FloatType * RESTRICT input_data, FloatType * RESTRICT output_data,
      const TensorIdx input_stride, const TensorIdx output_stride);

  template <GenNumType CONT,
            GenNumType NCONT>
  void cont_tiler_(DualCounter<CONT, NCONT>,
      const FloatType * RESTRICT input_data, FloatType * RESTRICT output_data,
      const TensorIdx input_stride, const TensorIdx output_stride);

  template <GenNumType NCONT>
  void cont_tiler_(DualCounter<0, NCONT>,
      const FloatType * RESTRICT input_data, FloatType * RESTRICT output_data,
      const TensorIdx input_stride, const TensorIdx output_stride);

  KernelFunc kernel_;
  GenNumType kn_wd_;    // Kernel width, number of elements in one register
};


template <typename FloatType>
class MacroTransMemcpy {
public:
  MacroTransMemcpy(DeducedFloatType<FloatType> alpha,
      DeducedFloatType<FloatType> beta);
  void operator()(const FloatType * RESTRICT input_data,
      FloatType * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride);

private:
  DeducedFloatType<FloatType> alpha, beta;
};


template <typename FloatType,
          CoefUsageTrans USAGE>
class MacroTransScalar {
public:
  MacroTransScalar(DeducedFloatType<FloatType> alpha,
      DeducedFloatType<FloatType> beta);
  void operator()(const FloatType * RESTRICT input_data,
      FloatType * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride);

private:
  DeducedFloatType<FloatType> alpha, beta;
};


/*
 * Alias and instantiation of vectorized macro kernels
 */
template <typename FloatType,
          CoefUsageTrans USAGE,
          GenNumType CONT_LEN,
          GenNumType NCONT_LEN>
using MacroTransVecFull = MacroTransVec<KernelTransFull<FloatType, USAGE>,
      CONT_LEN, NCONT_LEN>;


template <typename FloatType,
          CoefUsageTrans USAGE,
          GenNumType CONT_LEN,
          GenNumType NCONT_LEN>
using MacroTransVecHalf = MacroTransVec<KernelTransHalf<FloatType, USAGE>,
      CONT_LEN, NCONT_LEN>;


template <typename FloatType,
          CoefUsageTrans USAGE,
          GenNumType LEN>
using MacroTransLinear = MacroTransVec<KernelTransLinear<FloatType, USAGE>,
      LEN, 1>;


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


template <typename FloatType,
          CoefUsageTrans USAGE>
using MacroTransLinBig = MacroTransLinear<FloatType, USAGE, 8>;


template <typename FloatType,
          CoefUsageTrans USAGE>
using MacroTransLinMid = MacroTransLinear<FloatType, USAGE, 4>;


template <typename FloatType,
          CoefUsageTrans USAGE>
using MacroTransLinSmall = MacroTransLinear<FloatType, USAGE, 2>;


template <typename FloatType,
          CoefUsageTrans USAGE>
using MacroTransLinNano = MacroTransLinear<FloatType, USAGE, 1>;


/*
 * Import explicit template instantiation declaration
 */
#include "macro_kernel_trans.tcc"

}

#endif // HPTC_KERNELS_MACRO_KERNEL_TRANS_H_
