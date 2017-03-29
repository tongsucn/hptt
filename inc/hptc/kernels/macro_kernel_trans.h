#pragma once
#ifndef HPTC_KERNELS_MACRO_KERNEL_TRANS_H_
#define HPTC_KERNELS_MACRO_KERNEL_TRANS_H_

#include <hptc/types.h>
#include <hptc/util/util.h>
#include <hptc/util/util_trans.h>
#include <hptc/kernels/micro_kernel_trans.h>


namespace hptc {

template <typename MicroKernel,
          TensorUInt CONT_LEN,
          TensorUInt NCONT_LEN>
class MacroTrans {
public:
  using Float = typename MicroKernel::Float;

  MacroTrans();

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

  MicroKernel kernel_;
  const TensorUInt kn_width_;
};


template <typename FloatType>
class MacroTransLinear {
public:
  void set_coef(const DeducedFloatType<FloatType> alpha,
      const DeducedFloatType<FloatType> beta);

  void exec(const FloatType * RESTRICT input_data,
      FloatType * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride) const;

private:
  DeducedFloatType<FloatType> reg_alpha_, reg_beta_;
};


/*
 * Alias of macro kernels
 */
template <typename FloatType,
          TensorUInt CONT_LEN,
          TensorUInt NCONT_LEN>
using MacroTransFull = MacroTrans<KernelTransFull<FloatType>, CONT_LEN,
    NCONT_LEN>;


template <typename FloatType,
          TensorUInt CONT_LEN,
          TensorUInt NCONT_LEN>
using MacroTransHalf = MacroTrans<KernelTransHalf<FloatType>, CONT_LEN,
    NCONT_LEN>;


/*
 * Import explicit instantiation declaration
 */
#include "macro_kernel_trans.tcc"

}

#endif // HPTC_KERNELS_MACRO_KERNEL_TRANS_H_
