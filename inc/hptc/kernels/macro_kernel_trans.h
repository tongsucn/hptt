#pragma once
#ifndef HPTC_KERNELS_MACRO_KERNEL_TRANS_H_
#define HPTC_KERNELS_MACRO_KERNEL_TRANS_H_

#include <hptc/types.h>
#include <hptc/util/util.h>
#include <hptc/util/util_trans.h>
#include <hptc/kernels/micro_kernel_trans.h>


namespace hptc {

template <typename MicroKernel,
          TensorUInt SIZE_IN_INLD,
          TensorUInt SIZE_IN_OUTLD>
class MacroTrans {
public:
  using Float = typename MicroKernel::Float;

  MacroTrans();

  void set_coef(const DeducedFloatType<Float> alpha,
      const DeducedFloatType<Float> beta);

  TensorUInt get_cont_len() const;
  TensorUInt get_ncont_len() const;

  void exec(const Float *data_in, Float *data_out,
      const TensorIdx stride_in_outld, const TensorIdx stride_out_inld) const;

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

  MicroKernel kernel_;
  const TensorUInt kn_width_;
};


template <typename FloatType>
class MacroTransLinear {
public:
  static constexpr TensorUInt LOOP_MAX = KernelTransLinear<FloatType>::LOOP_MAX;

  void set_coef(const DeducedFloatType<FloatType> alpha,
      const DeducedFloatType<FloatType> beta);
  void set_wrapper_loop(const TensorIdx stride_in_inld,
      const TensorIdx stride_in_outld, const TensorIdx stride_out_inld,
      const TensorIdx stride_out_outld, const TensorUInt size_kn_inld,
      const TensorUInt size_kn_outld);

  void exec(const FloatType *data_in, FloatType *data_out,
      const TensorIdx size_trans, const TensorIdx size_pad) const;

private:
  KernelTransLinear<FloatType> kernel_;
};


template <typename FloatType>
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
          TensorUInt SIZE_IN_OUTLD>
using MacroTransFull = MacroTrans<KernelTransFull<FloatType>, SIZE_IN_INLD,
    SIZE_IN_OUTLD>;


template <typename FloatType,
          TensorUInt SIZE_IN_INLD,
          TensorUInt SIZE_IN_OUTLD>
using MacroTransHalf = MacroTrans<KernelTransHalf<FloatType>, SIZE_IN_INLD,
    SIZE_IN_OUTLD>;


/*
 * Import explicit instantiation declaration
 */
#include "macro_kernel_trans.tcc"

}

#endif // HPTC_KERNELS_MACRO_KERNEL_TRANS_H_
