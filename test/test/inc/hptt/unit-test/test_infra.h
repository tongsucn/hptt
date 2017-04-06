#pragma once
#ifndef HPTT_UNIT_TEST_TEST_INFRA_H_
#define HPTT_UNIT_TEST_TEST_INFRA_H_

#include <type_traits>

#include <gtest/gtest.h>

#include <hptt/tensor.h>
#include <hptt/types.h>

using namespace std;
using namespace hptt;


TEST(TestUtil, TestFloatTypeDeducer) {
  EXPECT_TRUE((is_same<float, DeducedFloatType<float>>::value))
      << "Float type deducing error, cannot deduce float to float.";
  EXPECT_TRUE((is_same<double, DeducedFloatType<double>>::value))
      << "Float type deducing error, cannot deduce double to double.";
  EXPECT_TRUE((is_same<float, DeducedFloatType<FloatComplex>>::value))
      << "Float type deducing error, cannot deduce FloatComplex to float.";
  EXPECT_TRUE((is_same<double, DeducedFloatType<DoubleComplex>>::value))
      << "Float type deducing error, cannot deduce DoubleComplex to double.";
}

#endif // HPTT_UNIT_TEST_TEST_INFRA_H_
