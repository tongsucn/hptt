#pragma once
#ifndef HPTC_UNIT_TEST_KERNELS_TEST_KERNEL_TRANS_AVX_H_
#define HPTC_UNIT_TEST_KERNELS_TEST_KERNEL_TRANS_AVX_H_

#include <random>
#include <algorithm>
#include <type_traits>

#include <gtest/gtest.h>
#include <immintrin.h>

#include <hptc/types.h>
#include <hptc/test_util.h>
#include <hptc/config/config_trans.h>
#include <hptc/kernels/kernel_trans.h>
#include <hptc/kernels/avx/kernel_trans_avx.h>

using namespace std;
using namespace hptc;


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
        act_data(new FloatType [this->data_len]) {
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

  virtual ~TestKernelTransAvx() {
    delete [] this->org_data;
    delete [] this->ref_data;
    delete [] this->act_data;
  }

  template <CoefUsageTrans USAGE>
  void calc_ref(TensorIdx offset_org_0, TensorIdx offset_org_1,
      TensorIdx offset_ref_0, TensorIdx offset_ref_1, GenNumType width) {
    // Reset reference data
    copy(this->org_data, this->org_data + this->data_len, this->ref_data);

    // Compute reference
    for (TensorIdx idx_0 = 0; idx_0 < width; ++idx_0) {
      for (TensorIdx idx_1 = 0; idx_1 < width; ++idx_1) {
        TensorIdx org_offset = (idx_0 + offset_org_0) * this->data_width
            + idx_1 + offset_org_1;
        TensorIdx ref_offset = idx_0 + offset_ref_0
            + (idx_1 + offset_ref_1) * this->data_width;

        if (CoefUsageTrans::USE_NONE == USAGE)
          this->ref_data[ref_offset] = this->org_data[org_offset];
        else if (CoefUsageTrans::USE_ALPHA == USAGE)
          this->ref_data[ref_offset] = this->alpha * this->org_data[org_offset];
        else if (CoefUsageTrans::USE_BETA == USAGE)
          this->ref_data[ref_offset] = this->org_data[org_offset]
              + this->beta * this->ref_data[ref_offset];
        else
          this->ref_data[ref_offset] = this->alpha * this->org_data[org_offset]
              + this->beta * this->ref_data[ref_offset];
      }
    }
  }

  void reset_act() {
    copy(this->org_data, this->org_data + this->data_len, this->act_data);
  }

  template <CoefUsageTrans USAGE,
            KernelTypeTrans TYPE>
  class CaseGenerator {
  public:
    CaseGenerator(TestKernelTransAvx<FloatType> &outer)
        : outer(outer) {
    }

    array<TensorIdx, 5> operator()() {
      // Create kernel, compute reference and reset actual data
      using KernelTypeTrans = KernelTransAvx<FloatType, USAGE, TYPE>;
      using RegType = typename KernelTypeTrans::RegType;

      KernelTransAvx<FloatType, USAGE, TYPE> kernel;
      RegType reg_alpha = kernel.reg_coef(outer.alpha);
      RegType reg_beta = kernel.reg_coef(outer.beta);
      array<TensorIdx, 5> result{ -1, 0, 0, 0, 0 };

      // Execute transpose
      for (TensorIdx &org_0 = result[1]; org_0 < outer.extra; ++org_0)
        for (TensorIdx &org_1 = result[2]; org_1 < outer.extra; ++org_1)
          for (TensorIdx &act_0 = result[3]; act_0 < outer.extra; ++act_0)
            for (TensorIdx &act_1 = result[4]; act_1 < outer.extra; ++act_1) {
              outer.calc_ref<USAGE>(org_0, org_1, act_0, act_1,
                  kernel.get_reg_num());
              outer.reset_act();
              kernel(outer.org_data + org_0 * outer.data_width + org_1,
                  outer.act_data + act_0 + act_1 * outer.data_width,
                  outer.data_width, outer.data_width, reg_alpha, reg_beta);
              result[0] = DataWrapper<FloatType>::verify(outer.ref_data,
                  outer.act_data, outer.data_len);
              if (-1 != result[0])
                return result;
            }

      return result;
    }

  private:
    TestKernelTransAvx<FloatType> &outer;
  };


  constexpr static TensorOrder extra = 2;
  constexpr static TensorOrder in_offset = sizeof(FloatType) / sizeof(Deduced);
  constexpr static Deduced alpha = static_cast<Deduced>(2.3f);
  constexpr static Deduced beta = static_cast<Deduced>(4.2f);

  GenNumType kernel_width_full, kernel_width_half;
  TensorIdx data_width, data_len;
  FloatType *org_data, *ref_data, *act_data;
};


using TestFloats = ::testing::Types<float, double, FloatComplex, DoubleComplex>;
TYPED_TEST_CASE(TestKernelTransAvx, TestFloats);


TYPED_TEST(TestKernelTransAvx, TestUtilities) {
  // Test register number selection
  // Full kernel
  KernelTransAvx<TypeParam, CoefUsageTrans::USE_BOTH,
      KernelTypeTrans::KERNEL_FULL> full;
  EXPECT_EQ(full.get_reg_num(), this->kernel_width_full)
      << "Register number does not match for full kernel.";

  // Half kernel
  KernelTransAvx<TypeParam, CoefUsageTrans::USE_BOTH,
      KernelTypeTrans::KERNEL_HALF> half;
  EXPECT_EQ(half.get_reg_num(), this->kernel_width_half)
      << "Register number does not match for half kernel.";
}


TYPED_TEST(TestKernelTransAvx, TestFullCoefNone) {
  typename TestKernelTransAvx<TypeParam>::CaseGenerator<
      CoefUsageTrans::USE_NONE, KernelTypeTrans::KERNEL_FULL> test_full(*this);
  auto result = test_full();
  ASSERT_EQ(-1, result[0]) << "Result of full kernel transpose without"
      << " coefficient does not match at absolute index: " << result[0]
      << ", offsets are: Original: (" << result[1] << ", " << result[2]
      << "), Actual: (" << result[3] << ", " << result[4] << ").";
}


TYPED_TEST(TestKernelTransAvx, TestFullCoefAlpha) {
  typename TestKernelTransAvx<TypeParam>::CaseGenerator<
      CoefUsageTrans::USE_ALPHA, KernelTypeTrans::KERNEL_FULL> test_full(*this);
  auto result = test_full();
  ASSERT_EQ(-1, result[0]) << "Result of full kernel transpose with alpha"
      << " does not match at absolute index: " << result[0]
      << ", offsets are: Original: (" << result[1] << ", " << result[2]
      << "), Actual: (" << result[3] << ", " << result[4] << ").";
}


TYPED_TEST(TestKernelTransAvx, TestFullCoefBeta) {
  typename TestKernelTransAvx<TypeParam>::CaseGenerator<
      CoefUsageTrans::USE_BETA, KernelTypeTrans::KERNEL_FULL> test_full(*this);
  auto result = test_full();
  ASSERT_EQ(-1, result[0]) << "Result of full kernel transpose with beta"
      << " does not match at absolute index: " << result[0]
      << ", offsets are: Original: (" << result[1] << ", " << result[2]
      << "), Actual: (" << result[3] << ", " << result[4] << ").";
}


TYPED_TEST(TestKernelTransAvx, TestFullCoefBoth) {
  typename TestKernelTransAvx<TypeParam>::CaseGenerator<
      CoefUsageTrans::USE_BOTH, KernelTypeTrans::KERNEL_FULL> test_full(*this);
  auto result = test_full();
  ASSERT_EQ(-1, result[0]) << "Result of full kernel transpose with both"
      << " coefficients does not match at absolute index: " << result[0]
      << ", offsets are: Original: (" << result[1] << ", " << result[2]
      << "), Actual: (" << result[3] << ", " << result[4] << ").";
}


TYPED_TEST(TestKernelTransAvx, TestHalfCoefNone) {
  typename TestKernelTransAvx<TypeParam>::CaseGenerator<
      CoefUsageTrans::USE_NONE, KernelTypeTrans::KERNEL_HALF> test_half(*this);
  auto result = test_half();
  ASSERT_EQ(-1, result[0]) << "Result of half kernel transpose without"
      << " coefficient does not match at absolute index: " << result[0]
      << ", offsets are: Original: (" << result[1] << ", " << result[2]
      << "), Actual: (" << result[3] << ", " << result[4] << ").";
}


TYPED_TEST(TestKernelTransAvx, TestHalfCoefAlpha) {
  typename TestKernelTransAvx<TypeParam>::CaseGenerator<
      CoefUsageTrans::USE_ALPHA, KernelTypeTrans::KERNEL_HALF> test_half(*this);
  auto result = test_half();
  ASSERT_EQ(-1, result[0]) << "Result of half kernel transpose with alpha"
      << " does not match at absolute index: " << result[0]
      << ", offsets are: Original: (" << result[1] << ", " << result[2]
      << "), Actual: (" << result[3] << ", " << result[4] << ").";
}


TYPED_TEST(TestKernelTransAvx, TestHalfCoefBeta) {
  typename TestKernelTransAvx<TypeParam>::CaseGenerator<
      CoefUsageTrans::USE_BETA, KernelTypeTrans::KERNEL_HALF> test_half(*this);
  auto result = test_half();
  ASSERT_EQ(-1, result[0]) << "Result of half kernel transpose with beta"
      << " does not match at absolute index: " << result[0]
      << ", offsets are: Original: (" << result[1] << ", " << result[2]
      << "), Actual: (" << result[3] << ", " << result[4] << ").";
}


TYPED_TEST(TestKernelTransAvx, TestHalfCoefBoth) {
  typename TestKernelTransAvx<TypeParam>::CaseGenerator<
      CoefUsageTrans::USE_BOTH, KernelTypeTrans::KERNEL_HALF> test_half(*this);
  auto result = test_half();
  ASSERT_EQ(-1, result[0]) << "Result of half kernel transpose with both"
      << " coefficients does not match at absolute index: " << result[0]
      << ", offsets are: Original: (" << result[1] << ", " << result[2]
      << "), Actual: (" << result[3] << ", " << result[4] << ").";
}

#endif // HPTC_UNIT_TEST_KERNELS_TEST_KERNEL_TRANS_AVX_H_
