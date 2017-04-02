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

  void exec(const Float *in_data, Float *out_data,
      const TensorIdx input_stride, const TensorIdx output_stride) const;

private:
  template <TensorUInt CONT,
            TensorUInt NCONT>
  void ncont_tiler_(DualCounter<CONT, NCONT>, const Float *in_data,
      Float *out_data, const TensorIdx input_stride,
      const TensorIdx output_stride) const;

  template <TensorUInt CONT>
  void ncont_tiler_(DualCounter<CONT, 0>, const Float *in_data,
      Float *out_data, const TensorIdx input_stride,
      const TensorIdx output_stride) const;

  template <TensorUInt CONT,
            TensorUInt NCONT>
  void cont_tiler_(DualCounter<CONT, NCONT>, const Float *in_data,
      Float *out_data, const TensorIdx input_stride,
      const TensorIdx output_stride) const;

  template <TensorUInt NCONT>
  void cont_tiler_(DualCounter<0, NCONT>, const Float *in_data,
      Float *out_data, const TensorIdx input_stride,
      const TensorIdx output_stride) const;

  MicroKernel kernel_;
  const TensorUInt kn_width_;
};


template <typename FloatType>
class MacroTransLinear {
public:
  static constexpr TensorUInt LOOP_MAX = 10;

  void set_coef(const DeducedFloatType<FloatType> alpha,
      const DeducedFloatType<FloatType> beta);
  void set_wrapper_loop(const TensorIdx stride_in_in,
      const TensorIdx stride_in_out, const TensorIdx stride_out_in,
      const TensorIdx stride_out_out, const TensorUInt ld_in_size,
      const TensorUInt ld_out_size);

  void exec(const FloatType *in_data, FloatType *out_data,
      const TensorIdx in_size, const TensorIdx out_size) const;

private:
  DeducedFloatType<FloatType> alpha_, beta_;
  TensorIdx stride_in_in_, stride_in_out_, stride_out_in_, stride_out_out_;
  TensorUInt ld_in_size_, ld_out_size_;
};


template <typename FloatType>
class MacroTransScalar {
public:
  void set_coef(const DeducedFloatType<FloatType> alpha,
      const DeducedFloatType<FloatType> beta);

  void exec(const FloatType *in_data, FloatType *out_data,
      const TensorIdx input_size, const TensorIdx output_size) const;

private:
  DeducedFloatType<FloatType> alpha_, beta_;
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
