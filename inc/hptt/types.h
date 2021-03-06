#pragma once
#ifndef HPTT_TYPES_H_
#define HPTT_TYPES_H_

#include <cstdint>
#include <ccomplex>


namespace hptt {

/*
 * Tensor assistant types
 */
using TensorIdx = uint64_t;
using TensorInt = int32_t;
using TensorUInt = uint32_t;


/*
 * Complex number types
 */
using FloatComplex = float _Complex;
using DoubleComplex = double _Complex;


/*
 * Coefficients type deducer.
 */
template <typename FloatType>
struct FloatTypeDeducer {
};

template <>
struct FloatTypeDeducer<float> {
  using type = float;
};

template <>
struct FloatTypeDeducer<FloatComplex> {
  using type = float;
};

template <>
struct FloatTypeDeducer<double> {
  using type = double;
};

template <>
struct FloatTypeDeducer<DoubleComplex> {
  using type = double;
};

template <typename FloatType>
using DeducedFloatType = typename FloatTypeDeducer<FloatType>::type;

}

#endif // HPTT_TYPES_H_
