#pragma once
#ifndef TEST_UNIT_TEST_TEST_TENSOR_WRAPPER_H_
#define TEST_UNIT_TEST_TEST_TENSOR_WRAPPER_H_

#include <array>

#include <gtest/gtest.h>

#include <hptc/tensor.h>
#include <hptc/types.h>

using namespace std;
using namespace hptc;


template <typename FloatType>
class TestTensorWrapper : public ::testing::Test {
protected:
  using Deduced = DeducedFloatType<FloatType>;

  TestTensorWrapper()
      : raw_data_len(256000),
        raw_data(new FloatType [256000]),
        size_order_1{ 3 },
        outer_size_order_1{ 9 },
        size_order_2{ 2, 15 },
        outer_size_order_2{ 7, 23 },
        size_order_4{ 20, 1, 32, 2 },
        outer_size_order_4{ 38, 1, 64, 54 },
        offsets_order_1{ 4 },
        offsets_order_2{ 3, 0 },
        offsets_order_4{ 8, 0, 31, 50 },
        tensor_col_order_1(this->size_order_1, this->raw_data),
        tensor_row_order_1(this->size_order_1, this->raw_data),
        tensor_col_order_4(this->size_order_4, this->raw_data),
        tensor_row_order_4(this->size_order_4, this->raw_data),
        sub_tensor_col_order_1(this->size_order_1, this->outer_size_order_1,
            this->offsets_order_1, this->raw_data),
        sub_tensor_row_order_1(this->size_order_1, this->outer_size_order_1,
            this->offsets_order_1, this->raw_data),
        sub_tensor_col_order_4(this->size_order_4, this->outer_size_order_4,
            this->offsets_order_4, this->raw_data),
        sub_tensor_row_order_4(this->size_order_4, this->outer_size_order_4,
            this->offsets_order_4, this->raw_data) {
    // Initialize raw data
    for (TensorIdx idx = 0; idx < this->raw_data_len; ++idx) {
      auto init_ptr = reinterpret_cast<Deduced *>(this->raw_data);
      for (TensorIdx inner_idx = 0; inner_idx < this->inner_offset; ++inner_idx)
        init_ptr[inner_idx] = static_cast<Deduced>(idx);
    }
  }

  ~TestTensorWrapper() {
    delete [] raw_data;
  }

  TensorIdx raw_data_len;
  FloatType *raw_data;
  static const TensorOrder inner_offset = sizeof(FloatType) / sizeof(Deduced);

  TensorSize<1> size_order_1, outer_size_order_1;
  TensorSize<2> size_order_2, outer_size_order_2;
  TensorSize<4> size_order_4, outer_size_order_4;
  array<TensorIdx, 1> offsets_order_1;
  array<TensorIdx, 2> offsets_order_2;
  array<TensorIdx, 4> offsets_order_4;

  TensorWrapper<FloatType, 1, MemLayout::COL_MAJOR> tensor_col_order_1;
  TensorWrapper<FloatType, 1, MemLayout::ROW_MAJOR> tensor_row_order_1;
  TensorWrapper<FloatType, 4, MemLayout::COL_MAJOR> tensor_col_order_4;
  TensorWrapper<FloatType, 4, MemLayout::ROW_MAJOR> tensor_row_order_4;
  TensorWrapper<FloatType, 1, MemLayout::COL_MAJOR> sub_tensor_col_order_1;
  TensorWrapper<FloatType, 1, MemLayout::ROW_MAJOR> sub_tensor_row_order_1;
  TensorWrapper<FloatType, 4, MemLayout::COL_MAJOR> sub_tensor_col_order_4;
  TensorWrapper<FloatType, 4, MemLayout::ROW_MAJOR> sub_tensor_row_order_4;
};


using TestFloats = ::testing::Types<float, double, FloatComplex, DoubleComplex>;
TYPED_TEST_CASE(TestTensorWrapper, TestFloats);


TYPED_TEST(TestTensorWrapper, TestTensorWrapperCreation) {
  // Prepare
  TensorSize<1> zero_size_order_1{ 0 };
  TensorSize<2> zero_size_order_2{ 0, 0 };
  TensorSize<4> zero_size_order_4{ 0, 0, 0, 0 };

  // Construction from default constructor
  TensorWrapper<TypeParam, 1, MemLayout::COL_MAJOR> def_tensor_col_order_1;
  ASSERT_EQ(zero_size_order_1, def_tensor_col_order_1.get_size())
      << "Default constructed 1-ord col-major tensor wrapper has nonzero size.";
  def_tensor_col_order_1.get_size() = this->size_order_1;
  ASSERT_EQ(this->size_order_1, def_tensor_col_order_1.get_size())
      << "Default constructed 1-ord col-major tensor wrapper cannot set size"
      << " correctly.";

  ASSERT_EQ(zero_size_order_1, def_tensor_col_order_1.get_outer_size())
      << "Default constructed 1-ord col-major tensor wrapper has nonzero outer"
      << " size.";
  def_tensor_col_order_1.get_outer_size() = this->outer_size_order_1;
  ASSERT_EQ(this->outer_size_order_1, def_tensor_col_order_1.get_outer_size())
      << "Default constructed 1-ord col-major tensor wrapper cannot set outer"
      << " size correctly.";

  ASSERT_EQ(nullptr, def_tensor_col_order_1.get_data())
      << "Default constructed 1-ord col-major tensor wrapper is initialized"
      << " with non-null pointer.";
  def_tensor_col_order_1.set_data(this->raw_data);
  ASSERT_EQ(this->raw_data, def_tensor_col_order_1.get_data())
      << "Default constructed 1-ord col-major tensor wrapper cannot set data"
      << " correctly.";


  TensorWrapper<TypeParam, 1, MemLayout::ROW_MAJOR> def_tensor_row_order_1;
  ASSERT_EQ(zero_size_order_1, def_tensor_row_order_1.get_size())
      << "Default constructed 1-ord row-major tensor wrapper has nonzero size.";
  def_tensor_row_order_1.get_size() = this->size_order_1;
  ASSERT_EQ(this->size_order_1, def_tensor_row_order_1.get_size())
      << "Default constructed 1-ord row-major tensor wrapper cannot set size"
      << " correctly.";

  ASSERT_EQ(zero_size_order_1, def_tensor_row_order_1.get_outer_size())
      << "Default constructed 1-ord row-major tensor wrapper has nonzero outer"
      << " size.";
  def_tensor_row_order_1.get_outer_size() = this->outer_size_order_1;
  ASSERT_EQ(this->outer_size_order_1, def_tensor_row_order_1.get_outer_size())
      << "Default constructed 1-ord row-major tensor wrapper cannot set outer"
      << " size correctly.";

  ASSERT_EQ(nullptr, def_tensor_row_order_1.get_data())
      << "Default constructed 1-ord row-major tensor wrapper is initialized"
      << " with non-null pointer.";
  def_tensor_row_order_1.set_data(this->raw_data);
  ASSERT_EQ(this->raw_data, def_tensor_row_order_1.get_data())
      << "Default constructed 1-ord row-major tensor wrapper cannot set data"
      << " correctly.";


  TensorWrapper<TypeParam, 2, MemLayout::COL_MAJOR> def_tensor_col_order_2;
  ASSERT_EQ(zero_size_order_2, def_tensor_col_order_2.get_size())
      << "Default constructed 2-ord col-major tensor wrapper has nonzero size.";
  def_tensor_col_order_2.get_size() = this->size_order_2;
  ASSERT_EQ(this->size_order_2, def_tensor_col_order_2.get_size())
      << "Default constructed 2-ord col-major tensor wrapper cannot set size"
      << " correctly.";

  ASSERT_EQ(zero_size_order_2, def_tensor_col_order_2.get_outer_size())
      << "Default constructed 2-ord col-major tensor wrapper has nonzero outer"
      << " size.";
  def_tensor_col_order_2.get_outer_size() = this->outer_size_order_2;
  ASSERT_EQ(this->outer_size_order_2, def_tensor_col_order_2.get_outer_size())
      << "Default constructed 2-ord col-major tensor wrapper cannot set outer"
      << " size correctly.";

  ASSERT_EQ(nullptr, def_tensor_col_order_2.get_data())
      << "Default constructed 2-ord col-major tensor wrapper is initialized"
      << " with non-null pointer.";
  def_tensor_col_order_2.set_data(this->raw_data);
  ASSERT_EQ(this->raw_data, def_tensor_col_order_2.get_data())
      << "Default constructed 2-ord col-major tensor wrapper cannot set data"
      << " correctly.";


  TensorWrapper<TypeParam, 2, MemLayout::ROW_MAJOR> def_tensor_row_order_2;
  ASSERT_EQ(zero_size_order_2, def_tensor_row_order_2.get_size())
      << "Default constructed 2-ord row-major tensor wrapper has nonzero size.";
  def_tensor_row_order_2.get_size() = this->size_order_2;
  ASSERT_EQ(this->size_order_2, def_tensor_row_order_2.get_size())
      << "Default constructed 2-ord row-major tensor wrapper cannot set size"
      << " correctly.";

  ASSERT_EQ(zero_size_order_2, def_tensor_row_order_2.get_outer_size())
      << "Default constructed 2-ord row-major tensor wrapper has nonzero outer"
      << " size.";
  def_tensor_row_order_2.get_outer_size() = this->outer_size_order_2;
  ASSERT_EQ(this->outer_size_order_2, def_tensor_row_order_2.get_outer_size())
      << "Default constructed 2-ord row-major tensor wrapper cannot set outer"
      << " size correctly.";

  ASSERT_EQ(nullptr, def_tensor_row_order_2.get_data())
      << "Default constructed 2-ord row-major tensor wrapper is initialized"
      << " with non-null pointer.";
  def_tensor_row_order_2.set_data(this->raw_data);
  ASSERT_EQ(this->raw_data, def_tensor_row_order_2.get_data())
      << "Default constructed 2-ord row-major tensor wrapper cannot set data"
      << " correctly.";


  TensorWrapper<TypeParam, 4, MemLayout::COL_MAJOR> def_tensor_col_order_4;
  ASSERT_EQ(zero_size_order_4, def_tensor_col_order_4.get_size())
      << "Default constructed 4-ord col-major tensor wrapper has nonzero size.";
  def_tensor_col_order_4.get_size() = this->size_order_4;
  ASSERT_EQ(this->size_order_4, def_tensor_col_order_4.get_size())
      << "Default constructed 4-ord col-major tensor wrapper cannot set size"
      << " correctly.";

  ASSERT_EQ(zero_size_order_4, def_tensor_col_order_4.get_outer_size())
      << "Default constructed 4-ord col-major tensor wrapper has nonzero outer"
      << " size.";
  def_tensor_col_order_4.get_outer_size() = this->outer_size_order_4;
  ASSERT_EQ(this->outer_size_order_4, def_tensor_col_order_4.get_outer_size())
      << "Default constructed 4-ord col-major tensor wrapper cannot set outer"
      << " size correctly.";

  ASSERT_EQ(nullptr, def_tensor_col_order_4.get_data())
      << "Default constructed 4-ord col-major tensor wrapper is initialized"
      << " with non-null pointer.";
  def_tensor_col_order_4.set_data(this->raw_data);
  ASSERT_EQ(this->raw_data, def_tensor_col_order_4.get_data())
      << "Default constructed 4-ord col-major tensor wrapper cannot set data"
      << " correctly.";


  TensorWrapper<TypeParam, 4, MemLayout::ROW_MAJOR> def_tensor_row_order_4;
  ASSERT_EQ(zero_size_order_4, def_tensor_row_order_4.get_size())
      << "Default constructed 4-ord row-major tensor wrapper has nonzero size.";
  def_tensor_row_order_4.get_size() = this->size_order_4;
  ASSERT_EQ(this->size_order_4, def_tensor_row_order_4.get_size())
      << "Default constructed 4-ord row-major tensor wrapper cannot set size"
      << " correctly.";

  ASSERT_EQ(zero_size_order_4, def_tensor_row_order_4.get_outer_size())
      << "Default constructed 4-ord row-major tensor wrapper has nonzero outer"
      << " size.";
  def_tensor_row_order_4.get_outer_size() = this->outer_size_order_4;
  ASSERT_EQ(this->outer_size_order_4, def_tensor_row_order_4.get_outer_size())
      << "Default constructed 4-ord row-major tensor wrapper cannot set outer"
      << " size correctly.";

  ASSERT_EQ(nullptr, def_tensor_row_order_4.get_data())
      << "Default constructed 4-ord row-major tensor wrapper is initialized"
      << " with non-null pointer.";
  def_tensor_row_order_4.set_data(this->raw_data);
  ASSERT_EQ(this->raw_data, def_tensor_row_order_4.get_data())
      << "Default constructed 4-ord row-major tensor wrapper cannot set data"
      << " correctly.";


  // Construction from user-defined constructor 1
  ASSERT_EQ(this->size_order_1, this->tensor_col_order_1.get_size())
      << "First constructor 1-ord col-major tensor wrapper does not have"
      << " expected size.";
  ASSERT_EQ(this->size_order_1, this->tensor_col_order_1.get_outer_size())
      << "First constructor 1-ord col-major tensor wrapper does not have"
      << " expected outer size.";
  ASSERT_EQ(this->raw_data, this->tensor_col_order_1.get_data())
      << "First constructor 1-ord col-major tensor wrapper does not have"
      << " correct data.";

  ASSERT_EQ(this->size_order_1, this->tensor_row_order_1.get_size())
      << "First constructor 1-ord row-major tensor wrapper does not have"
      << " expected size.";
  ASSERT_EQ(this->size_order_1, this->tensor_row_order_1.get_outer_size())
      << "First constructor 1-ord row-major tensor wrapper does not have"
      << " expected outer size.";
  ASSERT_EQ(this->raw_data, this->tensor_row_order_1.get_data())
      << "First constructor 1-ord row-major tensor wrapper does not have"
      << " correct data.";

  TensorWrapper<TypeParam, 2, MemLayout::COL_MAJOR> tensor_col_order_2(
      this->size_order_2, this->raw_data);
  ASSERT_EQ(this->size_order_2, tensor_col_order_2.get_size())
      << "First constructor 2-ord col-major tensor wrapper does not have"
      << " expected size.";
  ASSERT_EQ(this->size_order_2, tensor_col_order_2.get_outer_size())
      << "First constructor 2-ord col-major tensor wrapper does not have"
      << " expected outer size.";
  ASSERT_EQ(this->raw_data, tensor_col_order_2.get_data())
      << "First constructor 2-ord col-major tensor wrapper does not have"
      << " correct data.";

  TensorWrapper<TypeParam, 2, MemLayout::ROW_MAJOR> tensor_row_order_2(
      this->size_order_2, this->raw_data);
  ASSERT_EQ(this->size_order_2, tensor_row_order_2.get_size())
      << "First constructor 2-ord row-major tensor wrapper does not have"
      << " expected size.";
  ASSERT_EQ(this->size_order_2, tensor_row_order_2.get_outer_size())
      << "First constructor 2-ord row-major tensor wrapper does not have"
      << " expected outer size.";
  ASSERT_EQ(this->raw_data, tensor_row_order_2.get_data())
      << "First constructor 2-ord row-major tensor wrapper does not have"
      << " correct data.";

  ASSERT_EQ(this->size_order_4, this->tensor_col_order_4.get_size())
      << "First constructor 4-ord col-major tensor wrapper does not have"
      << " expected size.";
  ASSERT_EQ(this->size_order_4, this->tensor_col_order_4.get_outer_size())
      << "First constructor 4-ord col-major tensor wrapper does not have"
      << " expected outer size.";
  ASSERT_EQ(this->raw_data, this->tensor_col_order_4.get_data())
      << "First constructor 4-ord col-major tensor wrapper does not have"
      << " correct data.";

  ASSERT_EQ(this->size_order_4, this->tensor_row_order_4.get_size())
      << "First constructor 4-ord row-major tensor wrapper does not have"
      << " expected size.";
  ASSERT_EQ(this->size_order_4, this->tensor_row_order_4.get_outer_size())
      << "First constructor 4-ord row-major tensor wrapper does not have"
      << " expected outer size.";
  ASSERT_EQ(this->raw_data, this->tensor_row_order_4.get_data())
      << "First constructor 4-ord row-major tensor wrapper does not have"
      << " correct data.";


  // Construction from user-defined constructor 2
  ASSERT_EQ(this->size_order_1, this->sub_tensor_col_order_1.get_size())
      << "Second constructor 1-ord col-major tensor wrapper does not have"
      << " expected size.";
  ASSERT_EQ(this->outer_size_order_1,
      this->sub_tensor_col_order_1.get_outer_size())
      << "Second constructor 1-ord col-major tensor wrapper does not have"
      << " expected outer size.";
  ASSERT_EQ(this->raw_data, this->sub_tensor_col_order_1.get_data())
      << "Second constructor 1-ord col-major tensor wrapper does not have"
      << " correct data.";

  ASSERT_EQ(this->size_order_1, this->sub_tensor_row_order_1.get_size())
      << "Second constructor 1-ord row-major tensor wrapper does not have"
      << " expected size.";
  ASSERT_EQ(this->outer_size_order_1,
      this->sub_tensor_row_order_1.get_outer_size())
      << "Second constructor 1-ord row-major tensor wrapper does not have"
      << " expected outer size.";
  ASSERT_EQ(this->raw_data, this->sub_tensor_row_order_1.get_data())
      << "Second constructor 1-ord row-major tensor wrapper does not have"
      << " correct data.";

  TensorWrapper<TypeParam, 2, MemLayout::COL_MAJOR> sub_tensor_col_order_2(
      this->size_order_2, this->outer_size_order_2, this->offsets_order_2,
      this->raw_data);
  ASSERT_EQ(this->size_order_2, sub_tensor_col_order_2.get_size())
      << "Second constructor 2-ord col-major tensor wrapper does not have"
      << " expected size.";
  ASSERT_EQ(this->outer_size_order_2, sub_tensor_col_order_2.get_outer_size())
      << "Second constructor 2-ord col-major tensor wrapper does not have"
      << " expected outer size.";
  ASSERT_EQ(this->raw_data, sub_tensor_col_order_2.get_data())
      << "Second constructor 2-ord col-major tensor wrapper does not have"
      << " correct data.";

  TensorWrapper<TypeParam, 2, MemLayout::ROW_MAJOR> sub_tensor_row_order_2(
      this->size_order_2, this->outer_size_order_2, this->offsets_order_2,
      this->raw_data);
  ASSERT_EQ(this->size_order_2, sub_tensor_row_order_2.get_size())
      << "Second constructor 2-ord row-major tensor wrapper does not have"
      << " expected size.";
  ASSERT_EQ(this->outer_size_order_2, sub_tensor_row_order_2.get_outer_size())
      << "Second constructor 2-ord row-major tensor wrapper does not have"
      << " expected outer size.";
  ASSERT_EQ(this->raw_data, sub_tensor_row_order_2.get_data())
      << "Second constructor 2-ord row-major tensor wrapper does not have"
      << " correct data.";

  ASSERT_EQ(this->size_order_4, this->sub_tensor_col_order_4.get_size())
      << "Second constructor 4-ord col-major tensor wrapper does not have"
      << " expected size.";
  ASSERT_EQ(this->outer_size_order_4,
      this->sub_tensor_col_order_4.get_outer_size())
      << "Second constructor 4-ord col-major tensor wrapper does not have"
      << " expected outer size.";
  ASSERT_EQ(this->raw_data, this->sub_tensor_col_order_4.get_data())
      << "Second constructor 4-ord col-major tensor wrapper does not have"
      << " correct data.";

  ASSERT_EQ(this->size_order_4, this->sub_tensor_row_order_4.get_size())
      << "Second constructor 4-ord row-major tensor wrapper does not have"
      << " expected size.";
  ASSERT_EQ(this->outer_size_order_4,
      this->sub_tensor_row_order_4.get_outer_size())
      << "Second constructor 4-ord row-major tensor wrapper does not have"
      << " expected outer size.";
  ASSERT_EQ(this->raw_data, this->sub_tensor_row_order_4.get_data())
      << "Second constructor 4-ord row-major tensor wrapper does not have"
      << " correct data.";
}


TYPED_TEST(TestTensorWrapper, TestTensorWrapperIndexing) {
  // Col-major 1-order tensor wrapper indexing

  // Row-major 1-order tensor wrapper indexing

  // Col-major 4-order tensor wrapper indexing

  // Row-major 4-order tensor wrapper indexing

  // Col-major 1-order sub-tensor wrapper indexing

  // Row-major 1-order sub-tensor wrapper indexing

  // Col-major 4-order sub-tensor wrapper indexing

  // Row-major 4-order sub-tensor wrapper indexing

}


TYPED_TEST(TestTensorWrapper, TestTensorWrapperSlicing) {
  ;
}

#endif // TEST_UNIT_TEST_TEST_TENSOR_WRAPPER_H_
