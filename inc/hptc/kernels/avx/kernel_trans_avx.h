#pragma once
#ifndef HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_H_
#define HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_H_

#include <immintrin.h>

#include <type_traits>

#include <hptc/types.h>
#include <hptc/param/parameter_trans.h>


namespace hptc {

/*
 * Register type deducing utilities
 */
template <typename FloatType>
struct RegDeducer {
};

template <>
struct RegDeducer<float> {
  using type = __m256;
};

template <>
struct RegDeducer<double> {
  using type = __m256d;
};

template <>
struct RegDeducer<FloatComplex> {
  using type = __m256;
};

template <>
struct RegDeducer<DoubleComplex> {
  using type = __m256d;
};

template <typename FloatType>
using DeducedRegType = typename RegDeducer<FloatType>::type;


INLINE DeducedRegType<float> reg_coef(float coef);

INLINE DeducedRegType<double> reg_coef(double coef);


template <typename FloatType>
constexpr GenNumType reg_num_full();

template <typename FloatType>
constexpr GenNumType reg_num_half();


template <CoefUsage USAGE>
INLINE void kernel_trans_full_avx(const float * RESTRICT input_data,
    float * RESTRICT output_data, const TensorIdx input_stride,
    const TensorIdx output_stride, __m256 &reg_alpha, __m256 &reg_beta);


template <CoefUsage USAGE>
INLINE void kernel_trans_full_avx(const double * RESTRICT input_data,
    double * RESTRICT output_data, const TensorIdx input_stride,
    const TensorIdx output_stride, __m256d &reg_alpha, __m256d &reg_beta);


template <CoefUsage USAGE>
INLINE void kernel_trans_full_avx(const FloatComplex * RESTRICT input_data,
    FloatComplex * RESTRICT output_data, const TensorIdx input_stride,
    const TensorIdx output_stride, __m256 &reg_alpha, __m256 &reg_beta);


template <CoefUsage USAGE>
INLINE void kernel_trans_full_avx(const DoubleComplex * RESTRICT input_data,
    DoubleComplex * RESTRICT output_data, const TensorIdx input_stride,
    const TensorIdx output_stride, __m256d &reg_alpha, __m256d &reg_beta);


template <CoefUsage USAGE>
INLINE void kernel_trans_half_avx(const float * RESTRICT input_data,
    float * RESTRICT output_data, const TensorIdx input_stride,
    const TensorIdx output_stride, __m256 &reg_alpha, __m256 &reg_beta);


template <CoefUsage USAGE>
INLINE void kernel_trans_half_avx(const double * RESTRICT input_data,
    double * RESTRICT output_data, const TensorIdx input_stride,
    const TensorIdx output_stride, __m256d &reg_alpha, __m256d &reg_beta);


template <CoefUsage USAGE>
INLINE void kernel_trans_half_avx(const FloatComplex * RESTRICT input_data,
    FloatComplex * RESTRICT output_data, const TensorIdx input_stride,
    const TensorIdx output_stride, __m256 &reg_alpha, __m256 &reg_beta);


template <CoefUsage USAGE>
INLINE void kernel_trans_half_avx(const DoubleComplex * RESTRICT input_data,
    DoubleComplex * RESTRICT output_data, const TensorIdx input_stride,
    const TensorIdx output_stride, __m256d &reg_alpha, __m256d &reg_beta);


/*
 * Import implementation
 */
#include "kernel_trans_avx.tcc"

}

#endif // HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_H_
