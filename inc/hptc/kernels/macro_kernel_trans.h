#pragma once
#ifndef HPTC_KERNELS_MACRO_KERNEL_TRANS_H_
#define HPTC_KERNELS_MACRO_KERNEL_TRANS_H_

#include <memory>
#include <type_traits>

#include <hptc/types.h>
#include <hptc/param/parameter_trans.h>
#include <hptc/kernels/kernel_trans.h>


namespace hptc {

template <typename FloatType,
          typename KernelFunc,
          GenNumType CONT_LEN,
          GenNumType NCONT_LEN>
class MacroTransData {
public:
  MacroTransData(KernelFunc kernel, DeducedFloatType<FloatType> alpha,
      DeducedFloatType<FloatType> beta);

  INLINE GenNumType get_cont_len();
  INLINE GenNumType get_ncont_len();

protected:
  KernelFunc kernel_;
  DeducedRegType<FloatType> reg_alpha_, reg_beta_;
  GenNumType reg_num_;
};


template <typename FloatType,
          typename KernelFunc,
          GenNumType CONT_LEN,
          GenNumType NCONT_LEN>
class MacroTrans
    : public MacroTransData<FloatType, KernelFunc, CONT_LEN, NCONT_LEN> {
};


/*
 * Import implementation
 */
#include "macro_kernel_trans.tcc"

}

#endif // HPTC_KERNELS_MACRO_KERNEL_TRANS_H_
