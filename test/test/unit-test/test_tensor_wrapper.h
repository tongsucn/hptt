#pragma once
#ifndef TEST_TENSOR_WRAPPER_H_
#define TEST_TENSOR_WRAPPER_H_

#include <vector>

#include <gtest/gtest.h>

#include <hptc/tensor.h>
#include <hptc/types.h>

using namespace std;
using namespace hptc;


template <typename FloatType>
class TestTensorWrapper : public ::testing::Test {
protected:
  TestTensorWrapper()
      : test_num(4),
        size_obj{ { 9 }, { 7, 23 }, { 1, 23, 1 }, { 38, 1, 64, 54 } },
        sub_size_obj{ { 3 }, { 2, 15 }, { 1, 13, 1 }, { 20, 1, 32, 2 } },
        raw_data(this->test_num, nullptr),
        dim_offset{ { 3 }, { 3, 0 }, { 0, 10, 0 }, { 8, 0, 31, 50 } } {
    for (TensorIdx tensor_idx = 0; tensor_idx < this->test_num; ++tensor_idx) {
      // Initialize raw data
      TensorIdx total_size = 1;
      for (TensorIdx idx = 0; idx < this->size_obj[tensor_idx].get_dim(); ++idx)
        total_size *= this->size_obj[tensor_idx][idx];
      this->raw_data[tensor_idx] = new FloatType [total_size];
      for (TensorIdx ele_idx = 0; ele_idx < total_size; ++ele_idx)
        this->raw_data[tensor_idx][ele_idx] = static_cast<FloatType>(ele_idx);
    }
  }

  ~TestTensorWrapper() {
    for (auto ptr : this->raw_data)
      delete [] ptr;
  }

  const TensorIdx test_num;
  vector<TensorSize> size_obj, sub_size_obj;
  vector<FloatType *> raw_data;
  vector<vector<TensorIdx>> dim_offset;
};


using TestFloatTypes
    = ::testing::Types<float, double, FloatComplex, DoubleComplex>;
TYPED_TEST_CASE(TestTensorWrapper, TestFloatTypes);


TYPED_TEST(TestTensorWrapper, TestTensorWrapperCreation) {
  for (TensorIdx idx = 0; idx < this->test_num; ++idx) {
    // Creating original tensor from constructor
    TensorWrapper<TypeParam> tensor(this->size_obj[idx], this->raw_data[idx]);
    EXPECT_EQ(this->size_obj[idx], tensor.get_size())
        << "Size does not match in original tensor: " << idx;
    EXPECT_EQ(this->size_obj[idx], tensor.get_outer_size())
        << "Outer size does not match in original tensor: " << idx;
    ASSERT_EQ(this->raw_data[idx], tensor.get_data())
        << "Raw data does not match in original tensor: " << idx;

    // Creating sub-tensor from constructor
    TensorWrapper<TypeParam> sub_tensor(this->sub_size_obj[idx],
        this->size_obj[idx], this->dim_offset[idx], this->raw_data[idx]);
    EXPECT_EQ(this->sub_size_obj[idx], sub_tensor.get_size())
        << "Size does not match in sub-tensor: " << idx;
    EXPECT_EQ(this->size_obj[idx], sub_tensor.get_outer_size())
        << "Outer size does not match in sub-tensor: " << idx;
    ASSERT_EQ(this->raw_data[idx], sub_tensor.get_data())
        << "Raw data does not match in sub-tensor: " << idx;

    // Creating original tensor from copy
    TensorWrapper<TypeParam> copied_tensor(tensor);
    EXPECT_EQ(this->size_obj[idx], copied_tensor.get_size())
        << "Size does not match in copied original tensor: " << idx;
    EXPECT_EQ(this->size_obj[idx], copied_tensor.get_outer_size())
        << "Outer size does not match in copied original tensor: " << idx;
    ASSERT_EQ(this->raw_data[idx], copied_tensor.get_data())
        << "Raw data does not match in copied original tensor: " << idx;

    // Creating sub-tensor from copy constructor
    TensorWrapper<TypeParam> copied_sub_tensor(sub_tensor);
    EXPECT_EQ(this->sub_size_obj[idx], copied_sub_tensor.get_size())
        << "Size does not match in copied sub-tensor: " << idx;
    EXPECT_EQ(this->size_obj[idx], copied_sub_tensor.get_outer_size())
        << "Outer size does not match in copied sub-tensor: " << idx;
    ASSERT_EQ(this->raw_data[idx], copied_sub_tensor.get_data())
        << "Raw data does not match in copied sub-tensor: " << idx;

    // Creating original tensor from copy assignment
    TensorWrapper<TypeParam> &assigned_tensor = copied_sub_tensor;
    assigned_tensor = tensor;
    EXPECT_EQ(this->size_obj[idx], assigned_tensor.get_size())
        << "Size does not match in assigned original tensor: " << idx;
    EXPECT_EQ(this->size_obj[idx], assigned_tensor.get_outer_size())
        << "Outer size does not match in assigned original tensor: " << idx;
    ASSERT_EQ(this->raw_data[idx], assigned_tensor.get_data())
        << "Raw data does not match in assigned original tensor: " << idx;

    // Creating sub-tensor from copy assignment
    TensorWrapper<TypeParam> &assigned_sub_tensor = copied_tensor;
    assigned_sub_tensor = sub_tensor;
    EXPECT_EQ(this->sub_size_obj[idx], assigned_sub_tensor.get_size())
        << "Size does not match in assigned sub-tensor: " << idx;
    EXPECT_EQ(this->size_obj[idx], assigned_sub_tensor.get_outer_size())
        << "Outer size does not match in assigned sub-tensor: " << idx;
    ASSERT_EQ(this->raw_data[idx], assigned_sub_tensor.get_data())
        << "Raw data does not match in assigned sub-tensor: " << idx;
  }
}


TYPED_TEST(TestTensorWrapper, TestTensorWrapperIndexing) {
  // Test 1-dim original tensor indexing
  TensorWrapper<TypeParam> tensor_1_dim(this->size_obj[0], this->raw_data[0]);
  for (TensorIdx idx = -this->size_obj[0][0];
      idx < this->size_obj[0][0]; ++idx) {
    // Variadic-template-based indexing
    TensorIdx expect_val = idx < 0 ? this->size_obj[0][0] + idx : idx;
    ASSERT_EQ(static_cast<TypeParam>(expect_val), tensor_1_dim(idx))
        << "Template-based indexed content does not match at index: " << idx
        << ", raw data head address: " << tensor_1_dim.get_data()
        << ", element address: " << &tensor_1_dim(idx);

    // Vector-based indexing
    vector<TensorIdx> vec_idx{idx};
    ASSERT_EQ(expect_val, tensor_1_dim[vec_idx])
        << "Vector-based indexed content does not match at index: " << idx
        << ", raw data head address: " << tensor_1_dim.get_data()
        << ", element address: " << &tensor_1_dim[vec_idx];

    // Array-based indexing
    TensorIdx arr_idx[1] = {idx};
    ASSERT_EQ(expect_val, tensor_1_dim[arr_idx])
        << "Array-based indexed content does not match at index: " << idx
        << ", raw data head address: " << tensor_1_dim.get_data()
        << ", element address: " << &tensor_1_dim[arr_idx];
  }

  // Test 4-dim original tensor indexing
  TensorWrapper<TypeParam> tensor_4_dim(this->size_obj[3], this->raw_data[3]);
  TensorIdx expect_val = 0;
  for (TensorIdx idx_0 = 0; idx_0 < this->size_obj[3][0]; ++idx_0) {
    for (TensorIdx idx_1 = 0; idx_1 < this->size_obj[3][1]; ++idx_1) {
      for (TensorIdx idx_2 = 0; idx_2 < this->size_obj[3][2]; ++idx_2) {
        for (TensorIdx idx_3 = 0; idx_3 < this->size_obj[3][3]; ++idx_3) {
          // Variadic-template-based indexing
          ASSERT_EQ(static_cast<TypeParam>(expect_val),
              tensor_4_dim(idx_0, idx_1, idx_2, idx_3))
              << "Template-based indexed content does not match at index: ("
              << idx_0 << ", " << idx_1 << ", " << idx_2 << ", " << idx_3
              << "), raw data head address: " << tensor_4_dim.get_data()
              << ", element address: "
              << &tensor_4_dim(idx_0, idx_1, idx_2, idx_3);

          // Vector-based indexing
          vector<TensorIdx> vec_idx{ idx_0, idx_1, idx_2, idx_3 };
          ASSERT_EQ(static_cast<TypeParam>(expect_val), tensor_4_dim[vec_idx])
              << "Vector-based indexed content does not match at index: ("
              << idx_0 << ", " << idx_1 << ", " << idx_2 << ", " << idx_3
              << "), raw data head address: " << tensor_4_dim.get_data()
              << ", element address: " << &tensor_4_dim[vec_idx];

          // Array-based indexing
          TensorIdx arr_idx[4] = { idx_0, idx_1, idx_2, idx_3 };
          ASSERT_EQ(static_cast<TypeParam>(expect_val), tensor_4_dim[arr_idx])
              << "Array-based indexed content does not match at index: ("
              << idx_0 << ", " << idx_1 << ", " << idx_2 << ", " << idx_3
              << "), raw data head address: " << tensor_4_dim.get_data()
              << ", element address: " << &tensor_4_dim[arr_idx];

          ++expect_val;
        }
      }
    }
  }

  // Test 4-dim original tensor negative indexing
  expect_val = 0;
  for (TensorIdx idx_0 = -this->size_obj[3][0]; idx_0 < 0; ++idx_0) {
    for (TensorIdx idx_1 = -this->size_obj[3][1]; idx_1 < 0; ++idx_1) {
      for (TensorIdx idx_2 = -this->size_obj[3][2]; idx_2 < 0; ++idx_2) {
        for (TensorIdx idx_3 = -this->size_obj[3][3]; idx_3 < 0; ++idx_3) {
          // Variadic-template-based indexing
          ASSERT_EQ(static_cast<TypeParam>(expect_val),
              tensor_4_dim(idx_0, idx_1, idx_2, idx_3))
              << "Template-based neg-indexed content does not match at index: ("
              << idx_0 << ", " << idx_1 << ", " << idx_2 << ", " << idx_3
              << "), raw data head address: " << tensor_4_dim.get_data()
              << ", element address: "
              << &tensor_4_dim(idx_0, idx_1, idx_2, idx_3);

          // Vector-based indexing
          vector<TensorIdx> vec_idx{ idx_0, idx_1, idx_2, idx_3 };
          ASSERT_EQ(static_cast<TypeParam>(expect_val), tensor_4_dim[vec_idx])
              << "Vector-based neg-indexed content does not match at index: ("
              << idx_0 << ", " << idx_1 << ", " << idx_2 << ", " << idx_3
              << "), raw data head address: " << tensor_4_dim.get_data()
              << ", element address: " << &tensor_4_dim[vec_idx];

          // Array-based indexing
          TensorIdx arr_idx[4] = { idx_0, idx_1, idx_2, idx_3 };
          ASSERT_EQ(static_cast<TypeParam>(expect_val), tensor_4_dim[arr_idx])
              << "Array-based neg-indexed content does not match at index: ("
              << idx_0 << ", " << idx_1 << ", " << idx_2 << ", " << idx_3
              << "), raw data head address: " << tensor_4_dim.get_data()
              << ", element address: "<< &tensor_4_dim[arr_idx];

          ++expect_val;
        }
      }
    }
  }

  // Test 1-dim sub-tensor indexing
  TensorWrapper<TypeParam> sub_tensor_1_dim(this->sub_size_obj[0],
      this->size_obj[0], this->dim_offset[0], this->raw_data[0]);
  expect_val = this->dim_offset[0][0];
  for (TensorIdx idx = -this->sub_size_obj[0][0];
      idx < this->sub_size_obj[0][0]; ++idx) {
    if (0 == idx)
      expect_val = this->dim_offset[0][0];

    // Variadic-template-based indexing
    ASSERT_EQ(static_cast<TypeParam>(expect_val), sub_tensor_1_dim(idx))
        << "Template-based indexed content does not match at index: " << idx
        << ", raw data head address: " << sub_tensor_1_dim.get_data()
        << ", element address: " << &sub_tensor_1_dim(idx);

    // Vector-based indexing
    ASSERT_EQ(static_cast<TypeParam>(expect_val), sub_tensor_1_dim[{idx}])
        << "Vector-based indexed content does not match at index: " << idx
        << ", raw data head address: " << sub_tensor_1_dim.get_data()
        << ", element address: " << &sub_tensor_1_dim[{idx}];
    ++expect_val;
  }

  // Test 4-dim sub-tensor indexing
  TensorWrapper<TypeParam> sub_tensor_4_dim(this->sub_size_obj[3],
      this->size_obj[3], this->dim_offset[3], this->raw_data[3]);
  for (TensorIdx idx_0 = 0; idx_0 < this->sub_size_obj[3][0]; ++idx_0) {
    for (TensorIdx idx_1 = 0; idx_1 < this->sub_size_obj[3][1]; ++idx_1) {
      for (TensorIdx idx_2 = 0; idx_2 < this->sub_size_obj[3][2]; ++idx_2) {
        for (TensorIdx idx_3 = 0; idx_3 < this->sub_size_obj[3][3]; ++idx_3) {
          TensorIdx abs_idx_0 = idx_0 + this->dim_offset[3][0]
            + (idx_0 < 0 ? this->sub_size_obj[3][0] : 0);
          TensorIdx abs_idx_1 = idx_1 + this->dim_offset[3][1];
            + (idx_0 < 0 ? this->sub_size_obj[3][1] : 0);
          TensorIdx abs_idx_2 = idx_2 + this->dim_offset[3][2];
            + (idx_0 < 0 ? this->sub_size_obj[3][2] : 0);
          TensorIdx abs_idx_3 = idx_3 + this->dim_offset[3][3];
            + (idx_0 < 0 ? this->sub_size_obj[3][3] : 0);

          // Compute absolute offset
          TypeParam abs_expect = tensor_4_dim(abs_idx_0, abs_idx_1, abs_idx_2,
              abs_idx_3);

          // Variadic-template-based indexing
          ASSERT_EQ(abs_expect, sub_tensor_4_dim(idx_0, idx_1, idx_2, idx_3))
              << "Template-based indexed content does not match at index: ("
              << idx_0 << ", " << idx_1 << ", " << idx_2 << ", " << idx_3
              << "), raw data head address: "
              << sub_tensor_4_dim.get_data() << ", element address: "
              << &sub_tensor_4_dim(idx_0, idx_1, idx_2, idx_3);

          // Vector-based indexing
          vector<TensorIdx> vec_idx{ idx_0, idx_1, idx_2, idx_3 };
          ASSERT_EQ(abs_expect, sub_tensor_4_dim[vec_idx])
              << "Vector-based indexed content does not match at index: ("
              << idx_0 << ", " << idx_1 << ", " << idx_2 << ", " << idx_3
              << "), raw data head address: " << sub_tensor_4_dim.get_data()
              << ", element address: " << &sub_tensor_4_dim[vec_idx];

          // Array-based indexing
          TensorIdx arr_idx[4] = { idx_0, idx_1, idx_2, idx_3 };
          ASSERT_EQ(abs_expect, sub_tensor_4_dim[arr_idx])
              << "Array-based indexed content does not match at index: ("
              << idx_0 << ", " << idx_1 << ", " << idx_2 << ", " << idx_3
              << "), raw data head address: " << sub_tensor_4_dim.get_data()
              << ", element address: " << &sub_tensor_4_dim[arr_idx];
        }
      }
    }
  }
}


TYPED_TEST(TestTensorWrapper, TestTensorWrapperSlicing) {
  ;
}

#endif // TEST_TENSOR_WRAPPER_H_
