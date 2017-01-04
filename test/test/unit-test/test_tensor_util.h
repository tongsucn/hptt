#pragma once
#ifndef TEST_UNIT_TEST_TEST_TENSOR_UTIL_H_
#define TEST_UNIT_TEST_TEST_TENSOR_UTIL_H_

#include <array>

#include <gtest/gtest.h>

#include <hptc/tensor.h>
#include <hptc/types.h>

using namespace std;
using namespace hptc;


TEST(TestTensorUtil, TestTensorRangeIdxCreation) {
  // Prepare
  constexpr TensorIdx LOWER_BOUND = 0, UPPER_BOUND = 10;
  TRI range(LOWER_BOUND, UPPER_BOUND);

  // Test
  EXPECT_EQ(LOWER_BOUND, range.left_idx) << "Left index value does not match.";
  EXPECT_EQ(UPPER_BOUND, range.right_idx)
      << "Right index value does not match.";
}


TEST(TestTensorUtil, TestTensorSizeCreation) {
  // Prepare
  constexpr TensorOrder ORDER = 4;
  constexpr TensorOrder sizes[4] = { 12, 23, 34, 45 };

  // Test
  // Construction from default constructor
  TensorSize<ORDER> size_obj_default;
  for (TensorIdx idx = 0; idx < ORDER; ++idx)
    ASSERT_EQ(0, size_obj_default[idx])
        << "Default constructed size object does not provide zero size at dim-"
        << idx;

  // Construction from std::array
  array<TensorOrder, ORDER> arr{ sizes[0], sizes[1], sizes[2], sizes[3] };
  TensorSize<ORDER> size_obj_arr(arr);
  for (TensorIdx idx = 0; idx < ORDER; ++idx)
    ASSERT_EQ(arr[idx], size_obj_arr[idx])
        << "std::array constructed size object does not match at dim-" << idx;

  // Construction from initializer list
  TensorSize<ORDER> size_obj_list{ sizes[0], sizes[1], sizes[2], sizes[3] };
  for (TensorIdx idx = 0; idx < ORDER; ++idx)
    ASSERT_EQ(sizes[idx], size_obj_list[idx])
        << "Initializer list constructed size object does not match at dim-"
        << idx;
}


TEST(TestTensorUtil, TestTensorSizeCompare) {
  // Prepare
  constexpr TensorOrder ORDER = 4;
  constexpr array<TensorOrder, ORDER> sizes{ 12, 23, 34, 45 };
  TensorSize<ORDER> size_obj_0(sizes), size_obj_1(sizes);

  // Test
  for (TensorIdx idx = 0; idx < ORDER; ++idx) {
    EXPECT_TRUE(size_obj_0 == size_obj_1)
        << "Comparison does not provide expected result at dim-" << idx;
    ++size_obj_0[idx];
    EXPECT_FALSE(size_obj_0 == size_obj_1)
        << "Comparison does not provide expected result at dim-" << idx;
    --size_obj_0[idx];
  }
}

#endif // TEST_UNIT_TEST_TEST_TENSOR_UTIL_H_
