#pragma once
#ifndef HPTT_KERNELS_MACRO_KERNEL_TRANS_H_
#define HPTT_KERNELS_MACRO_KERNEL_TRANS_H_

#include <hptt/types.h>
#include <hptt/util/util.h>
#include <hptt/kernels/micro_kernel_trans.h>


namespace hptt {

template <typename MicroKernel,
          TensorUInt SIZE_IN_INLD,
          TensorUInt SIZE_IN_OUTLD>
class MacroTrans {
public:
  using Float = typename MicroKernel::Float;

  MacroTrans();

  TensorUInt get_cont_len() const;
  TensorUInt get_ncont_len() const;

  void set_coef(const DeducedFloatType<typename MicroKernel::Float> alpha,
      const DeducedFloatType<typename MicroKernel::Float> beta);

  void exec(const Float *data_in, Float *data_out,
      const TensorIdx stride_in_outld, const TensorIdx stride_out_inld) const;

protected:
  MicroKernel kernel_;
  const TensorUInt kn_width_;

private:
  template <TensorUInt IN_INLD,
            TensorUInt IN_OUTLD>
  void tile_outld_(DualCounter<IN_INLD, IN_OUTLD>, const Float *data_in,
      Float *data_out, const TensorIdx stride_in_outld,
      const TensorIdx stride_out_inld) const;

  template <TensorUInt IN_INLD>
  void tile_outld_(DualCounter<IN_INLD, 0>, const Float *data_in,
      Float *data_out, const TensorIdx stride_in_outld,
      const TensorIdx stride_out_inld) const;

  template <TensorUInt IN_INLD,
            TensorUInt IN_OUTLD>
  void tile_inld_(DualCounter<IN_INLD, IN_OUTLD>, const Float *data_in,
      Float *data_out, const TensorIdx stride_in_outld,
      const TensorIdx stride_out_inld) const;

  template <TensorUInt IN_OUTLD>
  void tile_inld_(DualCounter<0, IN_OUTLD>, const Float *data_in,
      Float *data_out, const TensorIdx stride_in_outld,
      const TensorIdx stride_out_inld) const;
};


template <typename FloatType,
          bool UPDATE_OUT>
class MacroTransLinear {
public:
  static constexpr TensorUInt LOOP_MAX = 10;

  MacroTransLinear();

  void set_coef(const DeducedFloatType<FloatType> alpha,
      const DeducedFloatType<FloatType> beta);
  void set_wrapper_loop(const TensorIdx stride_in_inld,
      const TensorIdx stride_in_outld, const TensorIdx stride_out_inld,
      const TensorIdx stride_out_outld, const TensorUInt size_kn_inld,
      const TensorUInt size_kn_outld);

  void exec(const FloatType *data_in, FloatType *data_out,
      const TensorIdx size_trans, const TensorIdx size_pad) const;

private:
  KernelTransLinear<FloatType, UPDATE_OUT> kernel_;

  TensorIdx stride_in_inld_, stride_in_outld_, stride_out_inld_,
      stride_out_outld_;
  TensorUInt size_kn_inld_, size_kn_outld_;
};


template <typename FloatType,
          bool UPDATE_OUT>
class MacroTransScalar {
public:
  void set_coef(const DeducedFloatType<FloatType> alpha,
      const DeducedFloatType<FloatType> beta);

  void exec(const FloatType *data_in, FloatType *data_out,
      const TensorIdx, const TensorIdx) const;

private:
  DeducedFloatType<FloatType> alpha_, beta_;
};


/*
 * Alias of macro kernels
 */
template <typename FloatType,
          TensorUInt SIZE_IN_INLD,
          TensorUInt SIZE_IN_OUTLD,
          bool UPDATE_OUT>
using MacroTransFull = MacroTrans<KernelTransFull<FloatType, UPDATE_OUT>,
    SIZE_IN_INLD, SIZE_IN_OUTLD>;


template <typename FloatType,
          TensorUInt SIZE_IN_INLD,
          TensorUInt SIZE_IN_OUTLD,
          bool UPDATE_OUT>
using MacroTransHalf = MacroTrans<KernelTransHalf<FloatType, UPDATE_OUT>,
    SIZE_IN_INLD, SIZE_IN_OUTLD>;


/*
 * Import explicit instantiation declaration
 */
#include "macro_kernel_trans.tcc"

}

#endif // HPTT_KERNELS_MACRO_KERNEL_TRANS_H_
