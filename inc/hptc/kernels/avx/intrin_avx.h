#pragma once
#ifndef HPTC_KERNELS_AVX_INTRIN_AVX_H_
#define HPTC_KERNELS_AVX_INTRIN_AVX_H_

#include <cstdint>
#include <xmmintrin.h>
#include <immintrin.h>

#include <memory>
#include <utility>

#include <hptc/types.h>
#include <hptc/util.h>
#include <hptc/kernels/kernel_trans_base.h>


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
struct RegTypeDeducer<FloatType> {
  using type = __m256;
};

template <>
struct RegTypeDeducer<DoubleType> {
  using type = __m256d;
};

template <typename FloatType>
using DeducedRegType = typename RegTypeDeducer<FloatType>::type;


template <GenNumType GEN_NUM,
          template Intrin,
          typename... Args>
INLINE void intrin_tiler(GenCounter<GEN_NUM>, Intrin intrinsic, Args... args);


template <typename Intrin,
          typename... Args>
INLINE void intrin_tiler(GenCounter<0>, Intrin intrinsic, Args... args);


template <typename DeducedType>
INLINE void intrin_load(GenNumType intrin_idx,
    const DeducedType * RESTRICT data, TensorIdx offset,
    DeducedRegType<DeducedType> * const reg);


template <typename DeducedType>
INLINE void intrin_store(GenNumType intrin_idx, DeducedType * RESTRICT data,
    TensorIdx offset, const DeducedRegType<DeducedType> * const reg);


template <typename DeducedType>
INLINE void intrin_set1(DeducedType val, DeducedRegType<DeducedType> *reg);


template <typename DeducedType>
INLINE void intrin_mul(GenNumType intrin_idx,
    DeducedRegType<DeducedType> * const reg_scaled,
    DeducedRegType<DeducedType> reg_coef);


template <typename DeducedType>
INLINE void intrin_add(GenNumType intrin_idx,
    DeducedRegType<DeducedType> * const reg_output,
    const DeducedRegType<DeducedType> * const reg_input);


/*
 * Import implementation
 */
#include "intrin_avx.tcc"

}

#endif // HPTC_KERNELS_AVX_INTRIN_AVX_H_
