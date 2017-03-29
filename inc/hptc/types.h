#pragma once
#ifndef HPTC_TYPES_H_
#define HPTC_TYPES_H_

#include <cstdint>
#include <ccomplex>

#include <hptc/compat.h>


namespace hptc {

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
 * Float type list
 */
enum class FloatList : TensorUInt {
  FLOAT             = 0x0,
  DOUBLE            = 0x1,
  FLOAT_COMPLEX     = 0x2,
  DOUBLE_COMPLEX    = 0x3
};


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

#endif // HPTC_TYPES_H_
