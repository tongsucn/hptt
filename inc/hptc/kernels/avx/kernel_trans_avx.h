#pragma once
#ifndef HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_H_
#define HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_H_

#include <cstdint>
#include <xmmintrin.h>
#include <immintrin.h>

#include <memory>
#include <utility>

#include <hptc/types.h>
#include <hptc/util.h>
#include <hptc/kernels/kernel_trans_base.h>
#include <hptc/kernels/avx/intrin_avx.h>


namespace hptc {

template <typename FloatType
          uint32_t REG_NUM = 1>
class KernelTransAvxBase : public KernelTransBase<FloatType> {
protected:
  DeducedRegType<FloatType> in_reg_arr[REG_NUM];
  DeducedRegType<FloatType> out_reg_arr[REG_NUM];

  virtual INLINE void in_reg_trans(const FloatType * RESTRICT input_data,
      TensorIdx input_offset) final;
  virtual INLINE void write_back(FloatType * RESTRICT output_data,
      TensorIdx output_offset) final;

  virtual INLINE void rescale_input(DeducedFloatType<FloatType> alpha) override;
  virtual INLINE void update_output(FloatType * RESTRICT output_data,
      TensorIdx output_offset, DeducedFloatType<FloatType> beta) override;
};


template <typename FloatType
          uint32_t REG_NUM>
class KernelTransAvx<FloatType, CoefUsage::USE_ALPHA, REG_NUM>
    : public KernelTransAvxBase<FloatType, REG_NUM> {
protected:
  virtual INLINE void rescale_input(DeducedFloatType<FloatType> alpha) final;
};


template <typename FloatType
          uint32_t REG_NUM>
class KernelTransAvx<FloatType, CoefUsage::USE_BETA, REG_NUM>
    : public KernelTransAvxBase<FloatType, REG_NUM> {
protected:
  virtual INLINE void update_output(FloatType * RESTRICT output_data,
      TensorIdx output_offset, DeducedFloatType<FloatType> beta) final;
};


template <typename FloatType
          uint32_t REG_NUM>
class KernelTransAvx<FloatType, CoefUsage::USE_BOTH, REG_NUM>
    : public KernelTransAvxBase<FloatType, REG_NUM> {
protected:
  virtual INLINE void rescale_input(DeducedFloatType<FloatType> alpha) final;
  virtual INLINE void update_output(FloatType * RESTRICT output_data,
      TensorIdx output_offset, DeducedFloatType<FloatType> beta) final;
};


/*
 * Import implementation
 */
#include "kernel_trans_avx.tcc"

}

#endif // HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_H_
