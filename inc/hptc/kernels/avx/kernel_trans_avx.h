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
          CoefUsage USAGE,
          KernelTransType KERNEL_TYPE>
class KernelTransAvx final
    : public KernelTransBase<FloatType, KERNEL_TYPE> {
};


template <CoefUsage USAGE>
class KernelTrans<float, USAGE, KernelTransType::KERNEL_FULL> final
    : public KernelTransBase<float, KernelTransType::KERNEL_FULL> {
public:
  KernelTransAvx(float alpha, float beta);

  KernelTransAvx(const KernelTransAvx &kernel) = delete;
  KernelTransAvx &operator=(const KernelTransAvx &kernel) = delete;
  ~KernelTransAvx() = default;

  virtual INLINE void operator()(const float * RESTRICT input_data,
      float * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride) final;

  virtual INLINE GenNumType get_reg_num() final;

private:
  __m256 reg_alpha, reg_beta;
};


template <CoefUsage USAGE>
class KernelTrans<double, USAGE, KernelTransType::KERNEL_FULL> final
    : public KernelTransBase<double, KernelTransType::KERNEL_FULL> {
public:
  KernelTransAvx(double alpha, double beta);

  KernelTransAvx(const KernelTransAvx &kernel) = delete;
  KernelTransAvx &operator=(const KernelTransAvx &kernel) = delete;
  ~KernelTransAvx() = default;

  virtual INLINE void operator()(const double * RESTRICT input_data,
      double * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride) final;

  virtual INLINE GenNumType get_reg_num() final;

private:
  __m256d reg_alpha, reg_beta;
};


template <CoefUsage USAGE>
class KernelTrans<FloatComplex, USAGE, KernelTransType::KERNEL_FULL> final
    : public KernelTransBase<FloatComplex, KernelTransType::KERNEL_FULL> {
public:
  KernelTransAvx(float alpha, float beta);

  KernelTransAvx(const KernelTransAvx &kernel) = delete;
  KernelTransAvx &operator=(const KernelTransAvx &kernel) = delete;
  ~KernelTransAvx() = default;

  virtual INLINE void operator()(const FloatComplex * RESTRICT input_data,
      FloatComplex * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride) final;

  virtual INLINE GenNumType get_reg_num() final;

private:
  __m256 reg_alpha, reg_beta;
};


template <CoefUsage USAGE>
class KernelTrans<DoubleComplex, USAGE, KernelTransType::KERNEL_FULL> final
    : public KernelTransBase<DoubleComplex, KernelTransType::KERNEL_FULL> {
public:
  KernelTransAvx(double alpha, double beta);

  KernelTransAvx(const KernelTransAvx &kernel) = delete;
  KernelTransAvx &operator=(const KernelTransAvx &kernel) = delete;
  ~KernelTransAvx() = default;

  virtual INLINE void operator()(const DoubleComplex * RESTRICT input_data,
      DoubleComplex * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride) final;

  virtual INLINE GenNumType get_reg_num() final;

private:
  __m256d reg_alpha, reg_beta;
};


template <CoefUsage USAGE>
class KernelTrans<float, USAGE, KernelTransType::KERNEL_HALF> final
    : public KernelTransBase<float, KernelTransType::KERNEL_HALF> {
public:
  KernelTransAvx(float alpha, float beta);

  KernelTransAvx(const KernelTransAvx &kernel) = delete;
  KernelTransAvx &operator=(const KernelTransAvx &kernel) = delete;
  ~KernelTransAvx() = default;

  virtual INLINE void operator()(const float * RESTRICT input_data,
      float * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride) final;

  virtual INLINE GenNumType get_reg_num() final;

private:
  __m256 reg_alpha, reg_beta;
};


template <CoefUsage USAGE>
class KernelTrans<double, USAGE, KernelTransType::KERNEL_HALF> final
    : public KernelTransBase<double, KernelTransType::KERNEL_HALF> {
public:
  KernelTransAvx(double alpha, double beta);

  KernelTransAvx(const KernelTransAvx &kernel) = delete;
  KernelTransAvx &operator=(const KernelTransAvx &kernel) = delete;
  ~KernelTransAvx() = default;

  virtual INLINE void operator()(const double * RESTRICT input_data,
      double * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride) final;

  virtual INLINE GenNumType get_reg_num() final;

private:
  __m256d reg_alpha, reg_beta;
};


template <CoefUsage USAGE>
class KernelTrans<FloatComplex, USAGE, KernelTransType::KERNEL_HALF> final
    : public KernelTransBase<FloatComplex, KernelTransType::KERNEL_HALF> {
public:
  KernelTransAvx(float alpha, float beta);

  KernelTransAvx(const KernelTransAvx &kernel) = delete;
  KernelTransAvx &operator=(const KernelTransAvx &kernel) = delete;
  ~KernelTransAvx() = default;

  virtual INLINE void operator()(const FloatComplex * RESTRICT input_data,
      FloatComplex * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride) final;

  virtual INLINE GenNumType get_reg_num() final;

private:
  __m256 reg_alpha, reg_beta;
};


template <CoefUsage USAGE>
class KernelTrans<DoubleComplex, USAGE, KernelTransType::KERNEL_HALF> final
    : public KernelTransBase<DoubleComplex, KernelTransType::KERNEL_HALF> {
public:
  KernelTransAvx(double alpha, double beta);

  KernelTransAvx(const KernelTransAvx &kernel) = delete;
  KernelTransAvx &operator=(const KernelTransAvx &kernel) = delete;
  ~KernelTransAvx() = default;

  virtual INLINE void operator()(const DoubleComplex * RESTRICT input_data,
      DoubleComplex * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride) final;

  virtual INLINE GenNumType get_reg_num() final;

private:
  __m256d reg_alpha, reg_beta;
};


/*
 * Import implementation
 */
#include "kernel_trans_avx.tcc"

}

#endif // HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_H_
