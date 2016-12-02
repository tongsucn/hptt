#pragma once
#ifndef HPTC_KERNELS_AVX_INTRIN_AVX_H_
#define HPTC_KERNELS_AVX_INTRIN_AVX_H_

#include <xmmintrin.h>
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


template <GenNumType GEN_NUM,
          typename Intrin,
          typename... Args>
INLINE void intrin_tiler(GenCounter<GEN_NUM>, Intrin intrinsic, Args... args);


template <typename Intrin,
          typename... Args>
INLINE void intrin_tiler(GenCounter<0>, Intrin intrinsic, Args... args);


template <typename DeducedType>
INLINE void intrin_load(GenNumType intrin_idx,
    const DeducedType * RESTRICT data, TensorIdx offset,
    DeducedRegType<DeducedType> reg[]);


template <typename DeducedType>
INLINE void intrin_store(GenNumType intrin_idx, DeducedType * RESTRICT data,
    TensorIdx offset, const DeducedRegType<DeducedType> reg[]);


template <typename DeducedType>
INLINE void intrin_set1(DeducedType val, DeducedRegType<DeducedType> *reg);


template <typename DeducedType>
INLINE void intrin_mul(GenNumType intrin_idx,
    DeducedRegType<DeducedType> reg_scaled[],
    DeducedRegType<DeducedType> reg_coef);


template <typename DeducedType>
INLINE void intrin_add(GenNumType intrin_idx,
    DeducedRegType<DeducedType> reg_output[],
    const DeducedRegType<DeducedType> reg_input[]);


template <typename FloatType>
INLINE void intrin_trans(DeducedRegType<FloatType> reg_input[]);

/*
 * Import implementation
 */
#include "intrin_avx.tcc"

}

#endif // HPTC_KERNELS_AVX_INTRIN_AVX_H_
