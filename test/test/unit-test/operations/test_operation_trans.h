#pragma once
#ifndef TEST_UNIT_TEST_OPERATIONS_TEST_OPERATION_TRANS_H_
#define TEST_UNIT_TEST_OPERATIONS_TEST_OPERATION_TRANS_H_

#include <vector>
#include <memory>

#include <gtest/gtest.h>

#include <hptc/types.h>
#include <hptc/tensor.h>
#include <hptc/param/parameter_trans.h>
#include <hptc/operations/operation_trans.h>

using namespace std;
using namespace hptc;


template <typename FloatType>
class TestOperationTrans : public ::testing::Test {
protected:
  using Tensor = TensorWrapper<FloatType>;
  using Deduced = DeducedFloatType<FloatType>;

  TestOperationTrans()
      : kernel_size(32 / sizeof(FloatType)),
        tensor_size(this->kernel_size * 4),
        data_len((this->tensor_size + 2) * (this->tensor_size + 2)),
        input_data(new FloatType [this->data_len]),
        output_data(new FloatType [this->data_len]),
        alpha(2.3), beta(4.2),
        size({ this->tensor_size, this->tensor_size }),
        outer_size({ this->tensor_size + 2, this->tensor_size + 2 }),
        input_tensor(this->size, this->outer_size, { 1, 0 }, this->input_data),
        output_tensor(this->size, this->outer_size, { 0, 2 },
            this->output_data),
        param(new ParamTrans<FloatType>(this->input_tensor, this->output_tensor,
            vector<TensorDim>({ 1, 0 }), this->alpha, this->beta)) {
    this->reset_data();
  }

  ~TestOperationTrans() {
    delete [] this->input_data;
    delete [] this->output_data;
  }

  void reset_data() {
    for (TensorIdx idx = 0; idx < data_len; ++idx) {
      this->set_ele(this->input_data, idx, idx);
      this->set_ele(this->output_data, idx, -1);
    }
  }

  void set_ele(FloatType *data, TensorIdx abs_offset, TensorIdx val) {
    FloatType &target = data[abs_offset];
    Deduced *set_ptr = reinterpret_cast<Deduced *>(&target);
    TensorIdx inner_offset = sizeof(FloatType) / sizeof(Deduced);
    for (TensorIdx idx = 0; idx < inner_offset; ++idx)
      set_ptr[idx] = static_cast<Deduced>(val);
  }

  void set_idx(TensorIdx row_idx, TensorIdx col_idx) {
    this->param->macro_loop_idx[0] = this->param->macro_loop_perm_idx[1]
        = row_idx;
    this->param->macro_loop_idx[1] = this->param->macro_loop_perm_idx[0]
        = col_idx;
  }


  TensorIdx kernel_size, tensor_size, data_len;
  FloatType *input_data, *output_data;
  TensorSize size, outer_size;
  Tensor input_tensor, output_tensor;
  const Deduced alpha, beta;
  shared_ptr<ParamTrans<FloatType>> param;
};


using FloatTypes = ::testing::Types<float, double, FloatComplex, DoubleComplex>;
TYPED_TEST_CASE(TestOperationTrans, FloatTypes);


TYPED_TEST(TestOperationTrans, MacroTransDefault) {
  using Tensor = TensorWrapper<TypeParam>;
  using Deduced = DeducedFloatType<TypeParam>;
  this->set_idx(0, 0);

  this->reset_data();
  OpMacroTrans4x3<TypeParam> macro_4x3(this->param);
  macro_4x3.exec();

  for (TensorIdx row_idx = 0; row_idx < 4 * this->kernel_size; ++row_idx) {
    for (TensorIdx col_idx = 0; col_idx < 3 * this->kernel_size; ++col_idx) {
      ASSERT_EQ(this->input_tensor(row_idx, col_idx),
          this->output_tensor(col_idx, row_idx))
          << "4x3 transpose result does not match expectation at row: "
          << row_idx << ", col: " << col_idx;
    }
  }

  this->reset_data();
  OpMacroTrans1x1<TypeParam> macro_1x1(this->param);
  macro_1x1.exec();

  for (TensorIdx row_idx = 0; row_idx < this->kernel_size; ++row_idx) {
    for (TensorIdx col_idx = 0; col_idx < this->kernel_size; ++col_idx) {
      ASSERT_EQ(this->input_tensor(row_idx, col_idx),
          this->output_tensor(col_idx, row_idx))
          << "1x1 transpose result does not match expectation at row: "
          << row_idx << ", col: " << col_idx;
    }
  }

  this->reset_data();
  OpMacroTrans2x2<TypeParam> macro_2x2(this->param);
  macro_2x2.exec();

  for (TensorIdx row_idx = 0; row_idx < 2 * this->kernel_size; ++row_idx) {
    for (TensorIdx col_idx = 0; col_idx < 2 * this->kernel_size; ++col_idx) {
      ASSERT_EQ(this->input_tensor(row_idx, col_idx),
          this->output_tensor(col_idx, row_idx))
          << "2x2 transpose result does not match expectation at row: "
          << row_idx << ", col: " << col_idx;
    }
  }

  this->reset_data();
  OpMacroTrans1x3<TypeParam> macro_1x3(this->param);
  macro_1x3.exec();

  for (TensorIdx row_idx = 0; row_idx < this->kernel_size; ++row_idx) {
    for (TensorIdx col_idx = 0; col_idx < 3 * this->kernel_size; ++col_idx) {
      ASSERT_EQ(this->input_tensor(row_idx, col_idx),
          this->output_tensor(col_idx, row_idx))
          << "1x3 transpose result does not match expectation at row: "
          << row_idx << ", col: " << col_idx;
    }
  }
}

#endif // TEST_UNIT_TEST_OPERATIONS_TEST_OPERATION_TRANS_H_
