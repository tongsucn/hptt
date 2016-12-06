#pragma once
#ifndef HPTC_KERNELS_KERNEL_TRANS_BASE_H_
#define HPTC_KERNELS_KERNEL_TRANS_BASE_H_

#include <hptc/types.h>


namespace hptc {

enum class CoefUsage : uint8_t {
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
  ~KernelTransBase() = default;

  virtual INLINE void operator()(
      const FloatType * RESTRICT input_data, FloatType * RESTRICT output_data,
      TensorIdx input_offset, TensorIdx output_offset,
      DeducedFloatType<FloatType> alpha, DeducedFloatType<FloatType> beta);

protected:
  virtual INLINE void in_reg_trans(const FloatType * RESTRICT input_data,
      TensorIdx input_offset) = 0;
  virtual INLINE void rescale_input(DeducedFloatType<FloatType> alpha) = 0;
  virtual INLINE void update_output(FloatType * RESTRICT output_data,
      TensorIdx output_offset, DeducedFloatType<FloatType> beta) = 0;
  virtual INLINE void write_back(FloatType * RESTRICT output_data,
      TensorIdx output_offset) = 0;
};


/*
 * Import implementation
 */
#include "kernel_trans_base.tcc"

}

#endif // HPTC_KERNELS_KERNEL_TRANS_BASE_H_
