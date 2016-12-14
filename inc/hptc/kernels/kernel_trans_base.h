#pragma once
#ifndef HPTC_KERNELS_KERNEL_TRANS_BASE_H_
#define HPTC_KERNELS_KERNEL_TRANS_BASE_H_

#include <hptc/types.h>


namespace hptc {

enum class CoefUsage : GenNumType {
  USE_NONE    = 0x0,
  USE_ALPHA   = 0x1,
  USE_BETA    = 0x2,
  USE_BOTH    = 0x3
};


template <typename FloatType>
class KernelTransBase {
public:
  KernelTransBase() = default;
  KernelTransBase(const KernelTransBase &kernel) = delete;
  KernelTransBase<FloatType> &operator=(const KernelTransBase &kernel) = delete;
  virtual ~KernelTransBase() = default;

  virtual void operator()(const FloatType * RESTRICT input_data,
      FloatType * RESTRICT output_data, TensorIdx input_stride,
      TensorIdx output_stride) = 0;

  virtual INLINE GenNumType get_reg_num() = 0;
};


/*
 * Import implementation
 */
#include "kernel_trans_base.tcc"

}

#endif // HPTC_KERNELS_KERNEL_TRANS_BASE_H_
