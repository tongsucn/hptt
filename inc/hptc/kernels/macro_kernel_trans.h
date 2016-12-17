#pragma once
#ifndef HPTC_KERNELS_MACRO_KERNEL_TRANS_H_
#define HPTC_KERNELS_MACRO_KERNEL_TRANS_H_

#include <memory>
#include <type_traits>

#include <hptc/types.h>
#include <hptc/compat.h>
#include <hptc/kernels/kernel_trans.h>


namespace hptc {

template <typename FloatType>
class MacroTransFunc {
public:
  virtual void operator()(const FloatType * RESTRICT input_data,
      FloatType * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride) = 0;
};


template <typename FloatType,
          typename KernelFunc>
class MacroTransData {
public:
  MacroTransData(KernelFunc kernel, DeduceFloatType<FloatType> alpha,
      DeduceFloatType<FloatType> beta);

protected:
  KernelFunc kernel_;
  DeducedRegType<FloatType> reg_alpha_, reg_beta_;
  GenNumType reg_num_;
};


template <typename FloatType,
          typename KernelFunc,
          MemLayout LAYOUT,
          GenNumType ROWS,
          GenNumType COLS>
class MacroTrans final
    : public MacroTransFunc<FloatType>,
      public MacroTransData<FloatType, KernelFunc> {
};


/*
 * Import implementation
 */
#include "macro_kernel_trans.tcc"

}

#endif // HPTC_KERNELS_MACRO_KERNEL_TRANS_H_
