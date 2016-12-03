#pragma once
#ifndef HPTC_KERNELS_AVX_INTRIN_AVX_H_
#define HPTC_KERNELS_AVX_INTRIN_AVX_H_

#include <immintrin.h>

#include <hptc/types.h>
#include <hptc/util.h>


namespace hptc {

/*
 * Register type deducer.
 */
template <typename FloatType>
struct RegTypeDeducer {
};

template <>
struct RegTypeDeducer<float> {
  using type = __m256;
};

template <>
struct RegTypeDeducer<double> {
  using type = __m256d;
};

template <>
struct RegTypeDeducer<FloatComplex> {
  using type = __m256;
};

template <>
struct RegTypeDeducer<DoubleComplex> {
  using type = __m256d;
};

template <typename FloatType>
using DeducedRegType = typename RegTypeDeducer<FloatType>::type;


template <typename DeducedType>
INLINE void intrin_avx_load(GenNumType intrin_idx,
    const DeducedType * RESTRICT data, TensorIdx offset,
    DeducedRegType<DeducedType> reg[]);


template <typename DeducedType>
INLINE void intrin_avx_store(GenNumType intrin_idx, DeducedType * RESTRICT data,
    TensorIdx offset, const DeducedRegType<DeducedType> reg[]);


template <typename DeducedType>
INLINE void intrin_avx_set1(DeducedType val, DeducedRegType<DeducedType> *reg);


template <typename DeducedType>
INLINE void intrin_avx_mul(GenNumType intrin_idx,
    DeducedRegType<DeducedType> reg_scaled[],
    DeducedRegType<DeducedType> reg_coef);


template <typename DeducedType>
INLINE void intrin_avx_add(GenNumType intrin_idx,
    DeducedRegType<DeducedType> reg_output[],
    const DeducedRegType<DeducedType> reg_input[]);


template <typename FloatType>
INLINE void intrin_avx_trans(DeducedRegType<FloatType> reg_input[]);

/*
 * Import implementation
 */
#include "intrin_avx.tcc"

}

#endif // HPTC_KERNELS_AVX_INTRIN_AVX_H_
