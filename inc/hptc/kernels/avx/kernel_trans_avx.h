#pragma once
#ifndef HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_H_
#define HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_H_

#include <immintrin.h>

#include <type_traits>
#include <iostream>

#include <hptc/types.h>
#include <hptc/kernels/kernel_trans_base.h>


namespace hptc {

template <typename FloatType,
          CoefUsage USAGE>
class KernelTransAvxFull final
    : public KernelTransBase<FloatType, KernelTransType::KERNEL_FULL> {
};


template <CoefUsage USAGE>
class KernelTransFull<float, USAGE> final
    : public KernelTransBase<float, KernelTransType::KERNEL_FULL> {
public:
  KernelTransAvxFull(float alpha, float beta);

  KernelTransAvxFull(const KernelTransAvxFull &kernel) = delete;
  KernelTransAvxFull &operator=(const KernelTransAvxFull &kernel) = delete;
  ~KernelTransAvxFull() = default;

  virtual INLINE void operator()(const float * RESTRICT input_data,
      float * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride) final;

  virtual INLINE GenNumType get_reg_num() final;

private:
  __m256 reg_alpha, reg_beta;
};


template <CoefUsage USAGE>
class KernelTransFull<double, USAGE> final
    : public KernelTransBase<double, KernelTransType::KERNEL_FULL> {
public:
  KernelTransAvxFull(double alpha, double beta);

  KernelTransAvxFull(const KernelTransAvxFull &kernel) = delete;
  KernelTransAvxFull &operator=(const KernelTransAvxFull &kernel) = delete;
  ~KernelTransAvxFull() = default;

  virtual INLINE void operator()(const double * RESTRICT input_data,
      double * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride) final;

  virtual INLINE GenNumType get_reg_num() final;

private:
  __m256d reg_alpha, reg_beta;
};


template <CoefUsage USAGE>
class KernelTransFull<FloatComplex, USAGE> final
    : public KernelTransBase<FloatComplex, KernelTransType::KERNEL_FULL> {
public:
  KernelTransAvxFull(float alpha, float beta);

  KernelTransAvxFull(const KernelTransAvxFull &kernel) = delete;
  KernelTransAvxFull &operator=(const KernelTransAvxFull &kernel) = delete;
  ~KernelTransAvxFull() = default;

  virtual INLINE void operator()(const FloatComplex * RESTRICT input_data,
      FloatComplex * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride) final;

  virtual INLINE GenNumType get_reg_num() final;

private:
  __m256 reg_alpha, reg_beta;
};


template <CoefUsage USAGE>
class KernelTransFull<DoubleComplex, USAGE> final
    : public KernelTransBase<DoubleComplex, KernelTransType::KERNEL_FULL> {
public:
  KernelTransAvxFull(double alpha, double beta);

  KernelTransAvxFull(const KernelTransAvxFull &kernel) = delete;
  KernelTransAvxFull &operator=(const KernelTransAvxFull &kernel) = delete;
  ~KernelTransAvxFull() = default;

  virtual INLINE void operator()(const DoubleComplex * RESTRICT input_data,
      DoubleComplex * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride) final;

  virtual INLINE GenNumType get_reg_num() final;

private:
  __m256d reg_alpha, reg_beta;
};


template <typename FloatType,
          CoefUsage USAGE>
class KernelTransAvxHalf final
    : public KernelTransBase<FloatType, KernelTransType::KERNEL_HALF> {
};


/*
 * Import implementation
 */
#include "kernel_trans_avx.tcc"

}

#endif // HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_H_
