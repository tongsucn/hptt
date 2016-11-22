#pragma once
#ifndef TEST_TENSOR_UTIL_H_
#define TEST_TENSOR_UTIL_H_

#include <cstdint>

#include <vector>

#include <gtest/gtest.h>

#include <hptc/tensor.h>

using namespace hptc;
using namespace std;

class TestTensorUtil : public ::testing::Test {
protected:
  TestTensorUtil()
    : RANGE_ZERO(0),
      RANGE_NONZERO(10),
      range(0, 10),
      size_obj_default(),
      size_obj_dim_zero(this->RANGE_ZERO),
      size_obj_dim_nonzero(this->RANGE_NONZERO),
      size_obj_lst_zero({}),
      size_obj_lst_nonzero({ 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 }) {
  }

  const TensorIdx RANGE_ZERO;
  const TensorIdx RANGE_NONZERO;
  TRI range;

  TensorSize size_obj_default;
  TensorSize size_obj_dim_zero;
  TensorSize size_obj_dim_nonzero;
  TensorSize size_obj_lst_zero;
  TensorSize size_obj_lst_nonzero;
};


TEST_F(TestTensorUtil, TestTensorRangeIdxCreation) {
  EXPECT_EQ(this->RANGE_ZERO, range.left_idx)
    << "Left index value does not match.";
  EXPECT_EQ(this->RANGE_NONZERO, range.right_idx)
    << "Right index value does not match.";
}


TEST_F(TestTensorUtil, TestTensorSizeCreation) {
  // Default construction
  EXPECT_EQ(this->RANGE_ZERO, this->size_obj_default.get_dim())
    << "Default constructor does not create a zero-dim size object.";

  // Construction from number of dimension
  EXPECT_EQ(this->RANGE_ZERO, this->size_obj_dim_zero.get_dim())
    << "Cannot create a " << this->RANGE_ZERO << "-dim size object by with-dim "
    << "constructor.";
  EXPECT_EQ(this->RANGE_NONZERO, this->size_obj_dim_nonzero.get_dim())
    << "Cannot create a " << this->RANGE_NONZERO << "-dim size object by with-"
    << "dim constructor.";

  // Construction from initializer list
  EXPECT_EQ(this->RANGE_ZERO, this->size_obj_lst_zero.get_dim())
    << "Cannot create a " << this->RANGE_ZERO << "-dim null size object from "
    << "initializer_list.";
  EXPECT_EQ(this->RANGE_NONZERO, this->size_obj_lst_nonzero.get_dim())
    << "Cannot create a " << this->RANGE_NONZERO << "-dim non-null size object "
    << "from initializer_list.";

  // Copy construction
  TensorSize size_obj_copy_zero(this->size_obj_lst_zero);
  EXPECT_EQ(this->RANGE_ZERO, size_obj_copy_zero.get_dim())
    << "Cannot create a " << this->RANGE_ZERO << "-dim null size object "
    << "from copy.";

  TensorSize size_obj_copy_nonzero(this->size_obj_lst_nonzero);
  EXPECT_EQ(this->RANGE_NONZERO, size_obj_copy_nonzero.get_dim())
    << "Cannot create a " << this->RANGE_NONZERO << "-dim non-null size object "
    << "from copy.";
  EXPECT_EQ(1, size_obj_copy_nonzero[this->RANGE_NONZERO - 1])
    << "Cannot access edge element after copy construction, target index: "
    << this->RANGE_NONZERO - 1;

  // Copy assignment
  size_obj_copy_nonzero = this->size_obj_lst_zero;
  EXPECT_EQ(this->RANGE_ZERO, size_obj_copy_nonzero.get_dim())
    << "Cannot create a " << this->RANGE_ZERO << "-dim null size object "
    << "from copy assignment.";

  size_obj_copy_zero = this->size_obj_lst_nonzero;
  EXPECT_EQ(this->RANGE_NONZERO, size_obj_copy_zero.get_dim())
    << "Cannot create a " << this->RANGE_NONZERO << "-dim non-null size object "
    << "from copy assignment.";
  EXPECT_EQ(1, size_obj_copy_zero[this->RANGE_NONZERO - 1])
    << "Cannot access edge element after copy assignment, target index: "
    << this->RANGE_NONZERO - 1;
}


TEST_F(TestTensorUtil, TestTensorSizeRandomAccess) {
  // Size object constructed from initializer list
  for (int32_t idx = -10; idx < 10; ++idx) {
    TensorIdx curr_val = this->size_obj_lst_nonzero[idx];
    TensorIdx expect_val = idx < 0 ? -idx : 10 - idx;
    EXPECT_EQ(expect_val, curr_val)
      << "Returned size value does not equal to expectation, target index: "
      << idx;
  }

  // Size object comparison
  EXPECT_TRUE(this->size_obj_lst_zero == this->size_obj_lst_zero)
    << "Comparison between the same 0-dim size object creates wrong result.";
  EXPECT_FALSE(this->size_obj_lst_nonzero == this->size_obj_lst_zero)
    << "Comparison between 0-dim and " << this->RANGE_NONZERO << "-dim size "
    << "object creates wrong result.";
  EXPECT_TRUE(this->size_obj_lst_nonzero == this->size_obj_lst_nonzero)
    << "Comparison between the same " << this->RANGE_NONZERO << "-dim size "
    << "object creates wrong result.";
  EXPECT_FALSE(this->size_obj_dim_nonzero == this->size_obj_lst_nonzero)
    << "Comparison between different " << this->RANGE_NONZERO << "-dim size "
    << "object creates wrong result.";
}

#endif
