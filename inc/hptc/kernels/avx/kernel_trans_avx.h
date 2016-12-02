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

template <typename FloatType,
          uint32_t REG_NUM = 1>
class KernelTransAvxBase : public KernelTransBase<FloatType> {
public:
  KernelTransAvxBase() = default;

  KernelTransAvxBase(const KernelTransAvxBase &kernel) = delete;
  KernelTransAvxBase<FloatType> &
  operator=(const KernelTransAvxBase &kernel) = delete;

  ~KernelTransAvxBase() = default;

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


template <typename FloatType,
          CoefUsage USAGE,
          uint32_t REG_NUM>
class KernelTransAvxImpl : public KernelTransAvxBase<FloatType, REG_NUM> {
};

template <typename FloatType,
          uint32_t REG_NUM>
class KernelTransAvxImpl<FloatType, CoefUsage::USE_ALPHA, REG_NUM>
    : public KernelTransAvxBase<FloatType, REG_NUM> {
protected:
  virtual INLINE void rescale_input(DeducedFloatType<FloatType> alpha) final;
};


template <typename FloatType,
          uint32_t REG_NUM>
class KernelTransAvxImpl<FloatType, CoefUsage::USE_BETA, REG_NUM>
    : public KernelTransAvxBase<FloatType, REG_NUM> {
protected:
  virtual INLINE void update_output(FloatType * RESTRICT output_data,
      TensorIdx output_offset, DeducedFloatType<FloatType> beta) final;
};


template <typename FloatType,
          uint32_t REG_NUM>
class KernelTransAvxImpl<FloatType, CoefUsage::USE_BOTH, REG_NUM>
    : public KernelTransAvxBase<FloatType, REG_NUM> {
protected:
  virtual INLINE void rescale_input(DeducedFloatType<FloatType> alpha) final;
  virtual INLINE void update_output(FloatType * RESTRICT output_data,
      TensorIdx output_offset, DeducedFloatType<FloatType> beta) final;
};


template <typename FloatType,
          CoefUsage USAGE,
          uint32_t REG_NUM>
class KernelTransAvx : public KernelTransAvxImpl<FloatType, USAGE, REG_NUM> {
};


template <CoefUsage USAGE>
class KernelTransAvx<float, USAGE, 0>
    : public KernelTransAvxImpl<float, USAGE, 8> {
};


template <CoefUsage USAGE>
class KernelTransAvx<double, USAGE, 0>
    : public KernelTransAvxImpl<double, USAGE, 4> {
};


template <CoefUsage USAGE>
class KernelTransAvx<FloatComplex, USAGE, 0>
    : public KernelTransAvxImpl<FloatComplex, USAGE, 4> {
};


template <CoefUsage USAGE>
class KernelTransAvx<DoubleComplex, USAGE, 0>
    : public KernelTransAvxImpl<DoubleComplex, USAGE, 2> {
};


template <typename FloatType,
          CoefUsage USAGE>
using KernelTransAvxDefault = KernelTransAvx<FloatType, USAGE, 0>;


/*
 * Import implementation
 */
#include "kernel_trans_avx.tcc"

}

#endif // HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_H_
