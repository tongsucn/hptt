#pragma once
#ifndef HPTC_TYPES_H_
#ifndef HPTC_TYPES_H_

#include <cstdint>
#include <ccomplex>

namespace hptc {

/*
 * Tensor assistant types
 */
using TensorIdx = int32_t;
using TensorDim = uint32_t;


/*
 * Complex number types
 */
using FloatComplex = float _Complex;
using DoubleComplex = double _Complex;
using LongDoubleComplex = long double _Complex;


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

template <>
struct FloatTypeDeducer<LongDoubleComplex> {
  using type = long double;
};

template <FloatType>
using CoefficientType = typename FloatTypeDeducer<FloatType>::type;

}

#endif // HPTC_TYPES_H_
