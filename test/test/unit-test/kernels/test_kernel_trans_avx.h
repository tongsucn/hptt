#pragma once
#ifndef TEST_UNIT_TEST_KERNELS_TEST_KERNEL_TRANS_AVX_H_
#define TEST_UNIT_TEST_KERNELS_TEST_KERNEL_TRANS_AVX_H_

#include <gtest/gtest.h>

#include <hptc/types.h>
#include <hptc/kernels/avx/kernel_trans_avx.h>

using namespace std;
using namespace hptc;


template <typename FloatType>
class TestKernelTransAvx : public ::testing::Test {
protected:
  using Deduced = DeducedFloatType<FloatType>;

  TestKernelTransAvx()
      : kernel_size(32 / sizeof(FloatType)),
        data_len(this->kernel_size * this->kernel_size),
        inner_offset(sizeof(FloatType) / sizeof(Deduced)),
        output_val(-1), alpha(2.3), beta(4.2),
        input_data(new FloatType [this->data_len]),
        output_data(new FloatType [this->data_len]),
        large_offset(this->kernel_size + 2),
        large_data_len(this->large_offset * this->large_offset),
        large_input_data(new FloatType [this->large_data_len]),
        large_output_data(new FloatType [this->large_data_len]) {
    this->reset_data();
  }

  ~TestKernelTransAvx() {
    delete [] this->input_data;
    delete [] this->output_data;
    delete [] this->large_input_data;
    delete [] this->large_output_data;
  }

  void reset_data() {
    for (TensorIdx idx = 0; idx < data_len; ++idx) {
      this->set_ele(this->input_data, idx, idx);
      this->set_ele(this->output_data, idx, this->output_val);
    }
    for (TensorIdx idx = 0; idx < large_data_len; ++idx) {
      this->set_ele(this->large_input_data, idx, idx);
      this->set_ele(this->large_output_data, idx, this->output_val);
    }
  }

  const FloatType &access_ele(const FloatType *data, TensorIdx offset,
      TensorIdx row_idx, TensorIdx col_idx) const {
    TensorIdx abs_offset = row_idx * offset + col_idx;
    return data[abs_offset];
  }

  FloatType &access_ele(FloatType *data, TensorIdx row_offset,
      TensorIdx col_offset, TensorIdx offset, TensorIdx row_idx,
      TensorIdx col_idx) {
    TensorIdx abs_offset = row_idx * offset + col_idx;
    return data[abs_offset];
  }

  void set_ele(FloatType *data, TensorIdx abs_offset, TensorIdx val) {
    FloatType &target = data[abs_offset];
    Deduced *set_ptr = reinterpret_cast<Deduced *>(&target);
    for (TensorIdx idx = 0; idx < this->inner_offset; ++idx)
      set_ptr[idx] = static_cast<Deduced>(val);
  }


  TensorIdx kernel_size, data_len, inner_offset;
  const TensorIdx output_val;
  Deduced alpha, beta;
  FloatType *input_data, *output_data;
  TensorIdx large_offset, large_data_len;
  FloatType *large_input_data, *large_output_data;
};


using FloatTypes = ::testing::Types<float, double, FloatComplex, DoubleComplex>;
TYPED_TEST_CASE(TestKernelTransAvx, FloatTypes);


TYPED_TEST(TestKernelTransAvx, TestDefaultKernelNonOffset) {
  using Deduced = DeducedFloatType<TypeParam>;
  const TypeParam *input = this->input_data;

  // Use both
  this->reset_data();
  KernelTransAvxDefault<TypeParam, CoefUsage::USE_BOTH> kernel_both;
  kernel_both(input, this->output_data, this->kernel_size, this->kernel_size,
      this->alpha, this->beta);

  for (TensorIdx row_idx = 0; row_idx < this->kernel_size; ++row_idx) {
    for (TensorIdx col_idx = 0; col_idx < this->kernel_size; ++col_idx) {
      TypeParam expect_val = access_ele(input, this->kernel_size, row_idx,
          col_idx);
      Deduced *ptr = reinterpret_cast<Deduced *>(&expect_val);
      for (TensorIdx idx = 0; idx < this->inner_offset; ++idx)
        ptr[idx] = ptr[idx] * this->alpha + this->output_val * this->beta;

      ASSERT_EQ(expect_val, access_ele(this->output_data, this->kernel_size,
              col_idx, row_idx))
          << "Transpose with alpha and beta result does not match at row idx: "
          << row_idx << ", col idx: " << col_idx;
    }
  }

  // Use alpha
  this->reset_data();
  KernelTransAvxDefault<TypeParam, CoefUsage::USE_ALPHA> kernel_alpha;
  kernel_alpha(input, this->output_data, this->kernel_size, this->kernel_size,
      this->alpha, 0.0);

  for (TensorIdx row_idx = 0; row_idx < this->kernel_size; ++row_idx) {
    for (TensorIdx col_idx = 0; col_idx < this->kernel_size; ++col_idx) {
      TypeParam expect_val = access_ele(input, this->kernel_size, row_idx,
          col_idx);
      Deduced *ptr = reinterpret_cast<Deduced *>(&expect_val);
      for (TensorIdx idx = 0; idx < this->inner_offset; ++idx)
        ptr[idx] = ptr[idx] * this->alpha;

      ASSERT_EQ(expect_val, access_ele(this->output_data, this->kernel_size,
              col_idx, row_idx))
          << "Transpose with alpha result does not match at row idx: "
          << row_idx << ", col idx: " << col_idx;
    }
  }

  // Use beta
  this->reset_data();
  KernelTransAvxDefault<TypeParam, CoefUsage::USE_BETA> kernel_beta;
  kernel_beta(input, this->output_data, this->kernel_size, this->kernel_size,
      1.0, this->beta);

  for (TensorIdx row_idx = 0; row_idx < this->kernel_size; ++row_idx) {
    for (TensorIdx col_idx = 0; col_idx < this->kernel_size; ++col_idx) {
      TypeParam expect_val = access_ele(input, this->kernel_size, row_idx,
          col_idx);
      Deduced *ptr = reinterpret_cast<Deduced *>(&expect_val);
      for (TensorIdx idx = 0; idx < this->inner_offset; ++idx)
        ptr[idx] += this->output_val * this->beta;

      ASSERT_EQ(expect_val, access_ele(this->output_data, this->kernel_size,
              col_idx, row_idx))
          << "Transpose with beta result does not match at row idx: "
          << row_idx << ", col idx: " << col_idx;
    }
  }

  // Use none
  this->reset_data();
  KernelTransAvxDefault<TypeParam, CoefUsage::USE_NONE> kernel_none;
  kernel_none(input, this->output_data, this->kernel_size, this->kernel_size,
      1.0, 0.0);

  for (TensorIdx row_idx = 0; row_idx < this->kernel_size; ++row_idx) {
    for (TensorIdx col_idx = 0; col_idx < this->kernel_size; ++col_idx) {
      ASSERT_EQ(access_ele(input, this->kernel_size, row_idx, col_idx),
          access_ele(this->output_data, this->kernel_size, col_idx, row_idx))
          << "Transpose with no coefficient result does not match at row idx: "
          << row_idx << ", col idx: " << col_idx;
    }
  }
}


TYPED_TEST(TestKernelTransAvx, TestDefaultKernelWithOffset) {
  using Deduced = DeducedFloatType<TypeParam>;

  for (TensorIdx row_offset = 0; row_offset <= 2; ++row_offset) {
    for (TensorIdx col_offset = 0; col_offset <= 2; ++col_offset) {
      TensorIdx sub_offset = row_offset * this->large_offset + col_offset;
      const TypeParam *input = this->large_input_data + sub_offset;
      TypeParam *output = this->large_output_data + sub_offset;

      // Use both
      this->reset_data();
      KernelTransAvxDefault<TypeParam, CoefUsage::USE_BOTH> kernel_both;
      kernel_both(input, output, this->large_offset, this->large_offset,
          this->alpha, this->beta);

      for (TensorIdx row_idx = 0; row_idx < this->kernel_size; ++row_idx) {
        for (TensorIdx col_idx = 0; col_idx < this->kernel_size; ++col_idx) {
          TypeParam expect_val = access_ele(input, this->large_offset, row_idx,
              col_idx);
          Deduced *ptr = reinterpret_cast<Deduced *>(&expect_val);
          for (TensorIdx idx = 0; idx < this->inner_offset; ++idx)
            ptr[idx] = ptr[idx] * this->alpha + this->output_val * this->beta;

          ASSERT_EQ(expect_val, access_ele(output, this->large_offset,
                  col_idx, row_idx))
              << "Transpose with alpha and beta result does not match at row "
              << "idx: " << row_idx << ", col idx: " << col_idx
              << ", row offset: " << row_offset << ", col_offset" << col_offset;
        }
      }

      // Use alpha
      this->reset_data();
      KernelTransAvxDefault<TypeParam, CoefUsage::USE_ALPHA> kernel_alpha;
      kernel_alpha(input, output, this->large_offset, this->large_offset,
          this->alpha, 0.0);

      for (TensorIdx row_idx = 0; row_idx < this->kernel_size; ++row_idx) {
        for (TensorIdx col_idx = 0; col_idx < this->kernel_size; ++col_idx) {
          TypeParam expect_val = access_ele(input, this->large_offset, row_idx,
              col_idx);
          Deduced *ptr = reinterpret_cast<Deduced *>(&expect_val);
          for (TensorIdx idx = 0; idx < this->inner_offset; ++idx)
            ptr[idx] = ptr[idx] * this->alpha;

          ASSERT_EQ(expect_val, access_ele(output, this->large_offset,
                  col_idx, row_idx))
              << "Transpose with alpha result does not match at row idx: "
              << row_idx << ", col idx: " << col_idx
              << ", row offset: " << row_offset << ", col_offset" << col_offset;
        }
      }

      // Use beta
      this->reset_data();
      KernelTransAvxDefault<TypeParam, CoefUsage::USE_BETA> kernel_beta;
      kernel_beta(input, output, this->large_offset, this->large_offset, 1.0,
          this->beta);

      for (TensorIdx row_idx = 0; row_idx < this->kernel_size; ++row_idx) {
        for (TensorIdx col_idx = 0; col_idx < this->kernel_size; ++col_idx) {
          TypeParam expect_val = access_ele(input, this->large_offset, row_idx,
              col_idx);
          Deduced *ptr = reinterpret_cast<Deduced *>(&expect_val);
          for (TensorIdx idx = 0; idx < this->inner_offset; ++idx)
            ptr[idx] += this->output_val * this->beta;

          ASSERT_EQ(expect_val, access_ele(output, this->large_offset,
                  col_idx, row_idx))
              << "Transpose with beta result does not match at row idx: "
              << row_idx << ", col idx: " << col_idx
              << ", row offset: " << row_offset << ", col_offset" << col_offset;
        }
      }

      // Use none
      this->reset_data();
      KernelTransAvxDefault<TypeParam, CoefUsage::USE_NONE> kernel_none;
      kernel_none(input, output, this->large_offset, this->large_offset, 1.0,
          0.0);

      for (TensorIdx row_idx = 0; row_idx < this->kernel_size; ++row_idx) {
        for (TensorIdx col_idx = 0; col_idx < this->kernel_size; ++col_idx) {
          ASSERT_EQ(access_ele(input, this->large_offset, row_idx, col_idx),
              access_ele(output, this->large_offset, col_idx, row_idx))
              << "Transpose with no coefficient result does not match at row idx: "
              << row_idx << ", col idx: " << col_idx
              << ", row offset: " << row_offset << ", col_offset" << col_offset;
        }
      }
    }
  }
}

#endif // TEST_UNIT_TEST_KERNELS_TEST_KERNEL_TRANS_AVX_H_
