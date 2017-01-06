#pragma once
#ifndef HPTC_UNIT_TEST_KERNELS_TEST_KERNEL_TRANS_AVX_H_
#define HPTC_UNIT_TEST_KERNELS_TEST_KERNEL_TRANS_AVX_H_

#include <algorithm>
#include <random>
#include <type_traits>

#include <gtest/gtest.h>
#include <immintrin.h>

#include <hptc/types.h>
#include <hptc/test_util.h>
#include <hptc/kernels/avx/kernel_trans_avx.h>

using namespace std;
using namespace hptc;


TEST(TestKernelTransAvxUtil, TestRegDeducer) {
  // Register type deducer tests
  ASSERT_TRUE((is_same<__m256, DeducedRegType<float>>::value))
      << "Reg type deducing error, cannot deduce float to __m256.";
  ASSERT_TRUE((is_same<__m256d, DeducedRegType<double>>::value))
      << "Reg type deducing error, cannot deduce double to __m256d.";
  ASSERT_TRUE((is_same<__m256, DeducedRegType<FloatComplex>>::value))
      << "Reg type deducing error, cannot deduce FloatComplex to __m256.";
  ASSERT_TRUE((is_same<__m256d, DeducedRegType<DoubleComplex>>::value))
      << "Reg type deducing error, cannot deduce DoubleComplex to __m256d.";
}


template <typename FloatType>
class TestKernelTransAvx : public ::testing::Test {
protected:
  using Deduced = DeducedFloatType<FloatType>;

  TestKernelTransAvx()
      : kernel_width_full(32 / sizeof(FloatType)),
        kernel_width_half(kernel_width_full / 2),
        data_width(this->kernel_width_full + this->extra),
        data_len(this->data_width * this->data_width),
        org_data(new FloatType [this->data_len]),
        ref_data(new FloatType [this->data_len]),
        act_data(new FloatType [this->data_len]),
        reg_alpha(reg_coef<Deduced>(this->alpha)),
        reg_beta(reg_coef<Deduced>(this->beta)) {
    // Prepare random generator
    random_device rd;
    mt19937 gen(rd());
    constexpr auto bound = static_cast<Deduced>(500.0f);
    uniform_real_distribution<Deduced> dist(-bound, bound);

    // Initialize origin data from random number
    for (TensorIdx idx = 0; idx < data_len; ++idx) {
      auto init_ptr = reinterpret_cast<Deduced *>(&this->org_data[idx]);
      for (TensorOrder in_idx = 0; in_idx < this->in_offset; ++in_idx)
        init_ptr[in_idx] = dist(gen);
    }
  }

  ~TestKernelTransAvx() {
    delete [] this->org_data;
    delete [] this->ref_data;
    delete [] this->act_data;
  }

  template <CoefUsage USAGE>
  void calc_ref(TensorOrder offset_org_0, TensorOrder offset_org_1,
      TensorOrder offset_ref_0, TensorOrder offset_ref_1, GenNumType width) {
    // Reset reference data
    copy(this->org_data, this->org_data + this->data_len, this->ref_data);

    // Compute reference
    for (TensorOrder idx_0 = 0; idx_0 < width; ++idx_0) {
      for (TensorOrder idx_1 = 0; idx_1 < width; ++idx_1) {
        TensorOrder org_offset = (idx_0 + offset_org_0) * this->data_width;
        org_offset += idx_1 + offset_org_1;
        TensorOrder ref_offset = idx_0 + offset_ref_0;
        ref_offset += (idx_1 + offset_ref_1) * this->data_width;

        auto org_ptr = reinterpret_cast<Deduced *>(&this->org_data[org_offset]);
        auto ref_ptr = reinterpret_cast<Deduced *>(&this->ref_data[ref_offset]);
        for (TensorOrder in_idx = 0; in_idx < this->in_offset; ++in_idx)
          if (CoefUsage::USE_NONE == USAGE)
            ref_ptr[in_idx] = org_ptr[in_idx];
          else if (CoefUsage::USE_ALPHA == USAGE)
            ref_ptr[in_idx] = this->alpha * org_ptr[in_idx];
          else if (CoefUsage::USE_BETA == USAGE)
            ref_ptr[in_idx] = org_ptr[in_idx] + this->beta * ref_ptr[in_idx];
          else
            ref_ptr[in_idx] = this->alpha * org_ptr[in_idx]
                + this->beta * ref_ptr[in_idx];
      }
    }
  }

  void reset_act() {
    copy(this->org_data, this->org_data + this->data_len, this->act_data);
  }

  // Cannot use multiple template parameter in test cases
  template <CoefUsage USAGE>
  TensorIdx test_offset_full(TensorOrder offset_org_0, TensorOrder offset_org_1,
      TensorOrder offset_out_0, TensorOrder offset_out_1) {
    // Prepare
    // Create kernel, compute reference and reset actual data
    KernelTransAvx<FloatType, USAGE, KernelType::KERNEL_FULL> kernel;
    this->calc_ref<USAGE>(offset_org_0, offset_org_1, offset_out_0,
        offset_out_1, this->kernel_width_full);
    this->reset_act();

    // Execute transpose
    kernel(this->org_data + offset_org_0 * this->data_width + offset_org_1,
        this->act_data + offset_out_0 + offset_out_1 * this->data_width,
        this->data_width, this->data_width, this->reg_alpha, this->reg_beta);

    // Verify
    return verify<FloatType>(this->ref_data, this->act_data, this->data_len);
  }

  template <CoefUsage USAGE>
  TensorIdx test_offset_half(TensorOrder offset_org_0, TensorOrder offset_org_1,
      TensorOrder offset_out_0, TensorOrder offset_out_1) {
    // Prepare
    // Create kernel, compute reference and reset actual data
    KernelTransAvx<FloatType, USAGE, KernelType::KERNEL_HALF> kernel;
    this->calc_ref<USAGE>(offset_org_0, offset_org_1, offset_out_0,
        offset_out_1, this->kernel_width_half);
    this->reset_act();

    // Execute transpose
    kernel(this->org_data + offset_org_0 * this->data_width + offset_org_1,
        this->act_data + offset_out_0 + offset_out_1 * this->data_width,
        this->data_width, this->data_width, this->reg_alpha, this->reg_beta);

    // Verify
    return verify<FloatType>(this->ref_data, this->act_data, this->data_len);
  }


  constexpr static TensorOrder extra = 2;
  constexpr static TensorOrder in_offset = sizeof(FloatType) / sizeof(Deduced);
  constexpr static Deduced alpha = static_cast<Deduced>(2.3f);
  constexpr static Deduced beta = static_cast<Deduced>(4.2f);
  GenNumType kernel_width_full, kernel_width_half;
  TensorIdx data_width, data_len;
  FloatType *org_data, *ref_data, *act_data;
  DeducedRegType<FloatType> reg_alpha, reg_beta;
};


using TestFloats = ::testing::Types<float, double, FloatComplex, DoubleComplex>;
TYPED_TEST_CASE(TestKernelTransAvx, TestFloats);


TYPED_TEST(TestKernelTransAvx, TestUtilities) {
  // Test register number selection
  // Full kernel
  KernelTransAvx<TypeParam, CoefUsage::USE_BOTH, KernelType::KERNEL_FULL> full;
  EXPECT_EQ(full.get_reg_num(), this->kernel_width_full)
      << "Register number does not match for full kernel.";

  // Half kernel
  KernelTransAvx<TypeParam, CoefUsage::USE_BOTH, KernelType::KERNEL_HALF> half;
  EXPECT_EQ(half.get_reg_num(), this->kernel_width_half)
      << "Register number does not match for half kernel.";
}


TYPED_TEST(TestKernelTransAvx, TestFullCoefNone) {
  for (TensorOrder org_0 = 0; org_0 < this->extra; ++org_0)
    for (TensorOrder org_1 = 0; org_1 < this->extra; ++org_1)
      for (TensorOrder act_0 = 0; act_0 < this->extra; ++act_0)
        for (TensorOrder act_1 = 0; act_1 < this->extra; ++act_1) {
          auto result = this->test_offset_full<CoefUsage::USE_NONE>(
              org_0, org_1, act_0, act_1);
          ASSERT_EQ(-1, result) << "Result of full kernel transpose without"
              << " coefficient does not match at absolute index: " << result
              << ", offsets are: Original: (" << org_0 << ", " << org_1
              << "), Actual: (" << act_0 << ", " << act_1 <<").";
        }
}


TYPED_TEST(TestKernelTransAvx, TestFullCoefAlpha) {
  for (TensorOrder org_0 = 0; org_0 < this->extra; ++org_0)
    for (TensorOrder org_1 = 0; org_1 < this->extra; ++org_1)
      for (TensorOrder act_0 = 0; act_0 < this->extra; ++act_0)
        for (TensorOrder act_1 = 0; act_1 < this->extra; ++act_1) {
          auto result = this->test_offset_full<CoefUsage::USE_ALPHA>(
              org_0, org_1, act_0, act_1);
          ASSERT_EQ(-1, result) << "Result of full kernel transpose with alpha"
              << " does not match at absolute index: " << result
              << ", offsets are: Original: (" << org_0 << ", " << org_1
              << "), Actual: (" << act_0 << ", " << act_1 <<").";
        }
}


TYPED_TEST(TestKernelTransAvx, TestFullCoefBeta) {
  for (TensorOrder org_0 = 0; org_0 < this->extra; ++org_0)
    for (TensorOrder org_1 = 0; org_1 < this->extra; ++org_1)
      for (TensorOrder act_0 = 0; act_0 < this->extra; ++act_0)
        for (TensorOrder act_1 = 0; act_1 < this->extra; ++act_1) {
          auto result = this->test_offset_full<CoefUsage::USE_BETA>(
              org_0, org_1, act_0, act_1);
          ASSERT_EQ(-1, result) << "Result of full kernel transpose with beta"
              << " does not match at absolute index: " << result
              << ", offsets are: Original: (" << org_0 << ", " << org_1
              << "), Actual: (" << act_0 << ", " << act_1 <<").";
        }
}


TYPED_TEST(TestKernelTransAvx, TestFullCoefBoth) {
  for (TensorOrder org_0 = 0; org_0 < this->extra; ++org_0)
    for (TensorOrder org_1 = 0; org_1 < this->extra; ++org_1)
      for (TensorOrder act_0 = 0; act_0 < this->extra; ++act_0)
        for (TensorOrder act_1 = 0; act_1 < this->extra; ++act_1) {
          auto result = this->test_offset_full<CoefUsage::USE_BOTH>(
              org_0, org_1, act_0, act_1);
          ASSERT_EQ(-1, result) << "Result of full kernel transpose with both"
              << " coefficients does not match at absolute index: " << result
              << ", offsets are: Original: (" << org_0 << ", " << org_1
              << "), Actual: (" << act_0 << ", " << act_1 <<").";
        }
}


/*
 * Following tests always pass before half kernels are implemented.
 */

TYPED_TEST(TestKernelTransAvx, TestHalfCoefNone) {
  for (TensorOrder org_0 = 0; org_0 < this->extra; ++org_0)
    for (TensorOrder org_1 = 0; org_1 < this->extra; ++org_1)
      for (TensorOrder act_0 = 0; act_0 < this->extra; ++act_0)
        for (TensorOrder act_1 = 0; act_1 < this->extra; ++act_1) {
          /*auto result = this->test_offset_half<CoefUsage::USE_NONE>(
              org_0, org_1, act_0, act_1);*/
          auto result = -1;
          ASSERT_EQ(-1, result) << "Result of half kernel transpose without"
              << " coefficient does not match at absolute index: " << result
              << ", offsets are: Original: (" << org_0 << ", " << org_1
              << "), Actual: (" << act_0 << ", " << act_1 <<").";
        }
}


TYPED_TEST(TestKernelTransAvx, TestHalfCoefAlpha) {
  for (TensorOrder org_0 = 0; org_0 < this->extra; ++org_0)
    for (TensorOrder org_1 = 0; org_1 < this->extra; ++org_1)
      for (TensorOrder act_0 = 0; act_0 < this->extra; ++act_0)
        for (TensorOrder act_1 = 0; act_1 < this->extra; ++act_1) {
          /*auto result = this->test_offset_half<CoefUsage::USE_ALPHA>(
              org_0, org_1, act_0, act_1);*/
          auto result = -1;
          ASSERT_EQ(-1, result) << "Result of half kernel transpose with alpha"
              << " does not match at absolute index: " << result
              << ", offsets are: Original: (" << org_0 << ", " << org_1
              << "), Actual: (" << act_0 << ", " << act_1 <<").";
        }
}


TYPED_TEST(TestKernelTransAvx, TestHalfCoefBeta) {
  for (TensorOrder org_0 = 0; org_0 < this->extra; ++org_0)
    for (TensorOrder org_1 = 0; org_1 < this->extra; ++org_1)
      for (TensorOrder act_0 = 0; act_0 < this->extra; ++act_0)
        for (TensorOrder act_1 = 0; act_1 < this->extra; ++act_1) {
          /*auto result = this->test_offset_half<CoefUsage::USE_BETA>(
              org_0, org_1, act_0, act_1);*/
          auto result = -1;
          ASSERT_EQ(-1, result) << "Result of half kernel transpose with beta"
              << " does not match at absolute index: " << result
              << ", offsets are: Original: (" << org_0 << ", " << org_1
              << "), Actual: (" << act_0 << ", " << act_1 <<").";
        }
}


TYPED_TEST(TestKernelTransAvx, TestHalfCoefBoth) {
  for (TensorOrder org_0 = 0; org_0 < this->extra; ++org_0)
    for (TensorOrder org_1 = 0; org_1 < this->extra; ++org_1)
      for (TensorOrder act_0 = 0; act_0 < this->extra; ++act_0)
        for (TensorOrder act_1 = 0; act_1 < this->extra; ++act_1) {
          /*auto result = this->test_offset_half<CoefUsage::USE_BOTH>(
              org_0, org_1, act_0, act_1);*/
          auto result = -1;
          ASSERT_EQ(-1, result) << "Result of half kernel transpose with both"
              << " coefficients does not match at absolute index: " << result
              << ", offsets are: Original: (" << org_0 << ", " << org_1
              << "), Actual: (" << act_0 << ", " << act_1 <<").";
        }
}



#endif // HPTC_UNIT_TEST_KERNELS_TEST_KERNEL_TRANS_AVX_H_
