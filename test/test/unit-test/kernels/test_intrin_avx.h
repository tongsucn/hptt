#pragma once
#ifndef TEST_UNIT_TEST_KERNELS_TEST_INTRIN_AVX_H_
#define TEST_UNIT_TEST_KERNELS_TEST_INTRIN_AVX_H_

#include <type_traits>

#include <gtest/gtest.h>

#include <hptc/types.h>
#include <hptc/util.h>
#include <hptc/kernels/avx/intrin_avx.h>

using namespace std;
using namespace hptc;


template <typename FloatType>
class TestIntrinSetAvx : public ::testing::Test {
protected:
  using Deduced = DeducedFloatType<FloatType>;
  TestIntrinSetAvx()
      : data_len(4 * 32 / sizeof(FloatType)),
        input_scale(2.3),
        output_scale(4.2),
        input_data(new FloatType [this->data_len]),
        output_data(new FloatType [this->data_len]),
        reg_num(32 / sizeof(FloatType)),
        mat_data(new FloatType [this->reg_num * 32 / sizeof(FloatType)]),
        inner_size(sizeof(FloatType) / sizeof(Deduced)),
        offset(32 / sizeof(Deduced)) {
    // Initialize data for basic intrinsics test
    for (TensorIdx idx = 0; idx < this->data_len; ++idx) {
      Deduced *input_init_ptr
          = reinterpret_cast<Deduced *>(&this->input_data[idx]);
      Deduced *output_init_ptr
          = reinterpret_cast<Deduced *>(&this->output_data[idx]);
      for (TensorIdx inner_idx = 0; inner_idx < this->inner_size; ++inner_idx) {
        *(input_init_ptr + inner_idx) = static_cast<Deduced>(idx);
        *(output_init_ptr + inner_idx) = static_cast<Deduced>(2);
      }
    }

    // Initialize data for transpose intrinsics test
    TensorIdx mat_len = this->reg_num * 32 / sizeof(FloatType);
    for (TensorIdx idx = 0; idx < mat_len; ++idx) {
      Deduced *init_ptr = reinterpret_cast<Deduced *>(&this->mat_data[idx]);
      for (TensorIdx inner_idx = 0; inner_idx < this->inner_size; ++inner_idx)
        *(init_ptr + inner_idx) = static_cast<Deduced>(idx);
    }
  }

  ~TestIntrinSetAvx() {
    delete [] this->input_data;
    delete [] this->output_data;
    delete [] this->mat_data;
  }

  TensorIdx data_len;
  Deduced input_scale, output_scale;
  FloatType *input_data;
  FloatType *output_data;
  TensorIdx reg_num;
  FloatType *mat_data;
  TensorIdx inner_size;
  TensorIdx offset;
};


using IntFloatTypes = ::testing::Types<float, double, FloatComplex, DoubleComplex>;
TYPED_TEST_CASE(TestIntrinSetAvx, IntFloatTypes);


TYPED_TEST(TestIntrinSetAvx, AvxBasic) {
  using Deduced = DeducedFloatType<TypeParam>;
  DeducedRegType<TypeParam> reg_input[4], reg_output[4], reg_input_scale,
      reg_output_scale;

  // Load data into eight 256-bits registers
  intrin_tiler(GenCounter<3>(), intrin_avx_load<Deduced>,
      reinterpret_cast<const Deduced *>(this->input_data), this->offset,
      reg_input);
  intrin_tiler(GenCounter<3>(), intrin_avx_load<Deduced>,
      reinterpret_cast<Deduced *>(this->output_data), this->offset, reg_output);

  // Load scale coefficient for input and ouput data
  intrin_avx_set1(this->input_scale, &reg_input_scale);
  intrin_avx_set1(this->output_scale, &reg_output_scale);

  // Rescale the content from input registers and output registers
  intrin_tiler(GenCounter<3>(), intrin_avx_mul<Deduced>, reg_input,
      reg_input_scale);
  intrin_tiler(GenCounter<3>(), intrin_avx_mul<Deduced>, reg_output,
      reg_output_scale);

  // Add the content from input and output registers into output registers
  intrin_tiler(GenCounter<3>(), intrin_avx_add<Deduced>, reg_output, reg_input);

  // Write back
  intrin_tiler(GenCounter<3>(), intrin_avx_store<Deduced>,
      reinterpret_cast<Deduced *>(this->output_data), this->offset, reg_output);

  // Verification
  for (TensorIdx idx = 0; idx < this->data_len; ++idx) {
    const Deduced *result_ptr
        = reinterpret_cast<const Deduced *>(&this->output_data[idx]);
    for (TensorIdx inner_idx = 0; inner_idx < this->inner_size; ++inner_idx) {
      ASSERT_EQ(*(result_ptr + inner_idx), static_cast<Deduced>(idx)
          * this->input_scale + 2 * this->output_scale)
          << "Result does not match at idx: " << idx << ", inner_idx: "
          << inner_idx;
    }
  }
}


TYPED_TEST(TestIntrinSetAvx, AvxTrans) {
  using Deduced = DeducedFloatType<TypeParam>;
  constexpr TensorIdx NUM = 32 / sizeof(TypeParam);
  DeducedRegType<TypeParam> reg_input[NUM];

  // Load, transpose and write back
  intrin_tiler(GenCounter<NUM>(), intrin_avx_load<Deduced>,
      reinterpret_cast<const Deduced *>(this->mat_data), this->offset,
      reg_input);
  intrin_avx_trans<TypeParam>(reg_input);
  intrin_tiler(GenCounter<NUM>(), intrin_avx_store<Deduced>,
      reinterpret_cast<Deduced *>(this->mat_data), this->offset, reg_input);

  // Verification
  TensorIdx abs_offset = 32 / sizeof(TypeParam);
  for (TensorIdx row_idx = 0; row_idx < NUM; ++row_idx) {
    for (TensorIdx col_idx = 0; col_idx < NUM; ++col_idx) {
      TensorIdx idx = row_idx * abs_offset + col_idx;
      TensorIdx prev_idx = col_idx * abs_offset + row_idx;
      const Deduced *result_ptr
          = reinterpret_cast<const Deduced *>(&this->mat_data[idx]);
      for (TensorIdx inner_idx = 0; inner_idx < this->inner_size; ++inner_idx) {
        ASSERT_EQ(*(result_ptr + inner_idx), static_cast<Deduced>(prev_idx))
            << "Result does not match at row_idx: " << row_idx << ", col_idx: "
            << col_idx << ", inner_idx: " << inner_idx;
      }
    }
  }
}


TEST(TestIntrinSetAvx, RegTypeDeducer) {
  EXPECT_TRUE((is_same<DeducedRegType<float>, __m256>::value))
    << "Incorrect register type deduce for type: float.";
  EXPECT_TRUE((is_same<DeducedRegType<double>, __m256d>::value))
    << "Incorrect register type deduce for type: double.";
  EXPECT_TRUE((is_same<DeducedRegType<FloatComplex>, __m256>::value))
    << "Incorrect register type deduce for type: FloatComplex.";
  EXPECT_TRUE((is_same<DeducedRegType<DoubleComplex>, __m256d>::value))
    << "Incorrect register type deduce for type: DoubleComplex.";
}


#endif // TEST_UNIT_TEST_KERNELS_TEST_INTRIN_AVX_H_
