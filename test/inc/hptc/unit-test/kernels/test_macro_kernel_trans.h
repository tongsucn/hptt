#pragma once
#ifndef HPTC_UNIT_TEST_KERNELS_TEST_MACRO_KERNEL_TRANS_H_
#define HPTC_UNIT_TEST_KERNELS_TEST_MACRO_KERNEL_TRANS_H_

#include <random>
#include <vector>
#include <algorithm>

#include <gtest/gtest.h>

#include <hptc/types.h>
#include <htpc/util.h>
#include <hptc/config/config_trans.h>
#include <hptc/kernels/kernel_trans.h>
#include <hptc/kernels/macro_kernel_trans.h>

using namespace std;
using namespace hptc;


template <typename FloatType>
class TestMacroTransBase {
protected:
  using Deduced = DeducedFloatType<FloatType>;

  TestMacroTransBase()
      : rd(), gen(this->rd()),
        dist(-500.0, 500.0) {
  }

  Deduced random_gen() {
    return this->dist(this->gen);
  }

  constexpr static TensorOrder in_offset = sizeof(FloatType) / sizeof(Deduced);
  constexpr static Deduced alpha = static_cast<Deduced>(2.3f);
  constexpr static Deduced beta = static_cast<Deduced>(4.2f);

  random_device rd;
  mt19937 gen;
  uniform_real_distribution<Deduced> dist;
};


template <typename FloatType>
class TestMacroTransVec
    : public TestMacroTransBase<FloatType>, public ::testing::Test {
protected:
  using Deduced = DeducedFloatType<FloatType>;

  TestMacroTransVec()
      : macro_width(this->kernel.get_reg_num() * this->max_macro),
        data_height(this->macro_width + this->height_extra),
        data_width(this->macro_width + this->width_extra),
        data_len(this->data_height * this->data_width),
        org_data(new FloatType [this->data_len]),
        ref_data(new FloatType [this->data_len]),
        act_data(new FloatType [this->data_len]) {
    // Initialize original data
    for (TensorIdx idx = 0; idx < this->data_len; ++idx) {
      auto init_ptr = reinterpret_cast<Deduced *>(&this->org_data[idx]);
      for (TensorOrder in_idx = 0; in_idx < this->in_offset; ++in_idx)
        init_ptr[in_idx] = this->random_gen();
    }
  }

  virtual ~TestMacroTransVec() {
    delete [] this->org_data;
    delete [] this->ref_data;
    delete [] this->act_data;
  }

  template <CoefUsageTrans USAGE>
  void calc_ref(TensorOrder offset_org_0, TensorOrder offset_org_1,
      TensorOrder offset_ref_0, TensorOrder offset_ref_1,
      TensorIdx rows, TensorIdx cols, TensorIdx kernel_width) {
    // Reset reference data
    copy(this->org_data, this->org_data + this->data_len, this->ref_data);

    // Compute reference
    for (TensorIdx idx_0 = 0; idx_0 < rows * kernel_width; ++idx_0) {
      for (TensorIdx idx_1 = 0; idx_1 < cols * kernel_width; ++idx_1) {
        TensorIdx org_offset = (idx_0 + offset_org_0) * this->data_width
            + idx_1 + offset_org_1;
        TensorIdx ref_offset = idx_0 + offset_ref_0
            + (idx_1 + offset_ref_1) * this->data_height;

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

  template <KernelTypeTrans TYPE,
            GenNumType HEIGHT,
            GenNumType WIDTH>
  class CaseGenerator {
  public:
    using Data = DataWrapper<FloatType>;

    CaseGenerator(TestMacroTransVec<FloatType> &outer)
        : outer(outer) {
    }

    array<TensorIdx, 8> operator()() {
      KernelTrans<FloatType, CoefUsageTrans::USE_NONE, TYPE> kernel_none;
      MacroTransVec<FloatType, decltype(kernel_none), WIDTH, HEIGHT>
          macro_none(outer.alpha, outer.beta);

      KernelTrans<FloatType, CoefUsageTrans::USE_ALPHA, TYPE> kernel_alpha;
      MacroTransVec<FloatType, decltype(kernel_alpha), WIDTH, HEIGHT>
          macro_alpha(outer.alpha, outer.beta);

      KernelTrans<FloatType, CoefUsageTrans::USE_BETA, TYPE> kernel_beta;
      MacroTransVec<FloatType, decltype(kernel_beta), WIDTH, HEIGHT>
          macro_beta(outer.alpha, outer.beta);

      KernelTrans<FloatType, CoefUsageTrans::USE_BOTH, TYPE> kernel_both;
      MacroTransVec<FloatType, decltype(kernel_both), WIDTH, HEIGHT>
          macro_both(outer.alpha, outer.beta);

      bool fail = false;
      array<TensorIdx, 8> res{ -1, -1, -1, -1, 0, 0, 0, 0 };
      for (TensorIdx &org_0 = res[4]; org_0 < outer.height_extra; ++org_0)
        for (TensorIdx &org_1 = res[4]; org_1 < outer.width_extra; ++org_1)
          for (TensorIdx &act_0 = res[6]; act_0 < outer.height_extra; ++act_0)
            for (TensorIdx &act_1 = res[7]; act_1 < outer.width_extra; ++act_1)
            {
              outer.calc_ref<CoefUsageTrans::USE_NONE>(org_0, org_1, act_0,
                  act_1, HEIGHT, WIDTH, kernel_none.get_reg_num());
              outer.reset_act();
              macro_none(outer.org_data + org_0 * outer.data_width + org_1,
                  outer.act_data + act_0 + act_1 * outer.data_height,
                  outer.data_width, outer.data_height);
              res[0] = Data::verify(outer.ref_data, outer.act_data,
                  outer.data_len);
              if (-1 != res[0])
                fail = true;

              outer.calc_ref<CoefUsageTrans::USE_ALPHA>(org_0, org_1, act_0,
                  act_1, HEIGHT, WIDTH, kernel_alpha.get_reg_num());
              outer.reset_act();
              macro_alpha(outer.org_data + org_0 * outer.data_width + org_1,
                  outer.act_data + act_0 + act_1 * outer.data_height,
                  outer.data_width, outer.data_height);
              res[1] = Data::verify(outer.ref_data, outer.act_data,
                  outer.data_len);
              if (-1 != res[1])
                fail = true;

              outer.calc_ref<CoefUsageTrans::USE_BETA>(org_0, org_1, act_0,
                  act_1, HEIGHT, WIDTH, kernel_beta.get_reg_num());
              outer.reset_act();
              macro_beta(outer.org_data + org_0 * outer.data_width + org_1,
                  outer.act_data + act_0 + act_1 * outer.data_height,
                  outer.data_width, outer.data_height);
              res[2] = Data::verify(outer.ref_data, outer.act_data,
                  outer.data_len);
              if (-1 != res[2])
                fail = true;

              outer.calc_ref<CoefUsageTrans::USE_BOTH>(org_0, org_1, act_0,
                  act_1, HEIGHT, WIDTH, kernel_both.get_reg_num());
              outer.reset_act();
              macro_both(outer.org_data + org_0 * outer.data_width + org_1,
                  outer.act_data + act_0 + act_1 * outer.data_height,
                  outer.data_width, outer.data_height);
              res[3] = Data::verify(outer.ref_data, outer.act_data,
                  outer.data_len);
              if (-1 != res[3])
                fail = true;

              if (fail)
                return res;
            }

      return res;
    }

  private:
    TestMacroTransVec<FloatType> &outer;
  };


  constexpr static GenNumType max_macro = 3;
  constexpr static TensorOrder height_extra = 3;
  constexpr static TensorOrder width_extra = 4;

  KernelTransFull<FloatType, CoefUsageTrans::USE_NONE> kernel;

  TensorIdx macro_width, data_height, data_width, data_len;
  FloatType *org_data, *ref_data, *act_data;
};


template <typename FloatType>
class TestMacroTransScalar
    : public TestMacroTransBase<FloatType>, public ::testing::Test {
protected:
  using Deduced = DeducedFloatType<FloatType>;
  using Data = DataWrapper<FloatType>;

  virtual void SetUp() {
    // Initialize origin data from random number
    auto init_ptr = reinterpret_cast<Deduced *>(&this->org_data);
    for (TensorOrder in_idx = 0; in_idx < this->in_offset; ++in_idx)
      init_ptr[in_idx] = this->random_gen();
  }

  template <CoefUsageTrans USAGE>
  void calc_ref() {
    // Reset reference data
    this->ref_data = this->org_data;

    // Compute reference data
    if (CoefUsageTrans::USE_NONE == USAGE)
      this->ref_data = this->org_data;
    else if (CoefUsageTrans::USE_ALPHA == USAGE)
      this->ref_data = this->alpha * this->org_data;
    else if (CoefUsageTrans::USE_BETA == USAGE)
      this->ref_data = this->org_data + this->beta * this->ref_data;
    else
      this->ref_data = this->alpha * this->org_data
          + this->beta * this->ref_data;
  }

  void reset_act() {
    this->act_data = this->org_data;
  }

  template <CoefUsageTrans USAGE>
  class CaseGenerator {
  public:
    CaseGenerator(TestMacroTransScalar<FloatType> &outer)
        : outer(outer) {
    }

    TensorIdx operator()() {
      MacroTransScalar<FloatType, USAGE> macro(outer.alpha, outer.beta);
      outer.calc_ref<USAGE>();
      outer.reset_act();

      // Execute transpose
      macro(&outer.org_data, &outer.act_data);

      // Verify
      return Data::verify(&outer.ref_data, &outer.act_data, 1);
    }

  private:
    TestMacroTransScalar<FloatType> &outer;
  };


  FloatType org_data, ref_data, act_data;
};


using TestFloats = ::testing::Types<float, double, FloatComplex, DoubleComplex>;
TYPED_TEST_CASE(TestMacroTransVec, TestFloats);
TYPED_TEST_CASE(TestMacroTransScalar, TestFloats);


TYPED_TEST(TestMacroTransVec, TestMacroFull1x1) {
  typename TestMacroTransVec<TypeParam>::CaseGenerator<
      KernelTypeTrans::KERNEL_FULL, 1, 1> test_full(*this);
  auto result = test_full();
  ASSERT_EQ(-1, result[0]) << "Result of 1x1 macro full kernel transpose"
      << " without coefficients does not match at absolute index: "
      << result[0] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";

  ASSERT_EQ(-1, result[1]) << "Result of 1x1 macro full kernel transpose"
      << " with alpha does not match at absolute index: "
      << result[1] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";

  ASSERT_EQ(-1, result[2]) << "Result of 1x1 macro full kernel transpose"
      << " with beta does not match at absolute index: "
      << result[2] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";

  ASSERT_EQ(-1, result[3]) << "Result of 1x1 macro full kernel transpose"
      << " with both coefficients does not match at absolute index: "
      << result[3] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";
}


TYPED_TEST(TestMacroTransVec, TestMacroFull1x2) {
  typename TestMacroTransVec<TypeParam>::CaseGenerator<
      KernelTypeTrans::KERNEL_FULL, 1, 2> test_full(*this);
  auto result = test_full();
  ASSERT_EQ(-1, result[0]) << "Result of 1x2 macro full kernel transpose"
      << " without coefficients does not match at absolute index: "
      << result[0] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";

  ASSERT_EQ(-1, result[1]) << "Result of 1x2 macro full kernel transpose"
      << " with alpha does not match at absolute index: "
      << result[1] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";

  ASSERT_EQ(-1, result[2]) << "Result of 1x2 macro full kernel transpose"
      << " with beta does not match at absolute index: "
      << result[2] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";

  ASSERT_EQ(-1, result[3]) << "Result of 1x2 macro full kernel transpose"
      << " with both coefficients does not match at absolute index: "
      << result[3] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";
}


TYPED_TEST(TestMacroTransVec, TestMacroFull1x3) {
  typename TestMacroTransVec<TypeParam>::CaseGenerator<
      KernelTypeTrans::KERNEL_FULL, 1, 3> test_full(*this);
  auto result = test_full();
  ASSERT_EQ(-1, result[0]) << "Result of 1x3 macro full kernel transpose"
      << " without coefficients does not match at absolute index: "
      << result[0] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";

  ASSERT_EQ(-1, result[1]) << "Result of 1x3 macro full kernel transpose"
      << " with alpha does not match at absolute index: "
      << result[1] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";

  ASSERT_EQ(-1, result[2]) << "Result of 1x3 macro full kernel transpose"
      << " with beta does not match at absolute index: "
      << result[2] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";

  ASSERT_EQ(-1, result[3]) << "Result of 1x3 macro full kernel transpose"
      << " with both coefficients does not match at absolute index: "
      << result[3] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";
}


TYPED_TEST(TestMacroTransVec, TestMacroFull2x1) {
  typename TestMacroTransVec<TypeParam>::CaseGenerator<
      KernelTypeTrans::KERNEL_FULL, 2, 1> test_full(*this);
  auto result = test_full();
  ASSERT_EQ(-1, result[0]) << "Result of 2x1 macro full kernel transpose"
      << " without coefficients does not match at absolute index: "
      << result[0] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";

  ASSERT_EQ(-1, result[1]) << "Result of 2x1 macro full kernel transpose"
      << " with alpha does not match at absolute index: "
      << result[1] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";

  ASSERT_EQ(-1, result[2]) << "Result of 2x1 macro full kernel transpose"
      << " with beta does not match at absolute index: "
      << result[2] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";

  ASSERT_EQ(-1, result[3]) << "Result of 2x1 macro full kernel transpose"
      << " with both coefficients does not match at absolute index: "
      << result[3] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";
}


TYPED_TEST(TestMacroTransVec, TestMacroFull2x3) {
  typename TestMacroTransVec<TypeParam>::CaseGenerator<
      KernelTypeTrans::KERNEL_FULL, 2, 3> test_full(*this);
  auto result = test_full();
  ASSERT_EQ(-1, result[0]) << "Result of 2x3 macro full kernel transpose"
      << " without coefficients does not match at absolute index: "
      << result[0] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";

  ASSERT_EQ(-1, result[1]) << "Result of 2x3 macro full kernel transpose"
      << " with alpha does not match at absolute index: "
      << result[1] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";

  ASSERT_EQ(-1, result[2]) << "Result of 2x3 macro full kernel transpose"
      << " with beta does not match at absolute index: "
      << result[2] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";

  ASSERT_EQ(-1, result[3]) << "Result of 2x3 macro full kernel transpose"
      << " with both coefficients does not match at absolute index: "
      << result[3] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";
}


TYPED_TEST(TestMacroTransVec, TestMacroFull3x1) {
  typename TestMacroTransVec<TypeParam>::CaseGenerator<
      KernelTypeTrans::KERNEL_FULL, 3, 1> test_full(*this);
  auto result = test_full();
  ASSERT_EQ(-1, result[0]) << "Result of 3x1 macro full kernel transpose"
      << " without coefficients does not match at absolute index: "
      << result[0] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";

  ASSERT_EQ(-1, result[1]) << "Result of 3x1 macro full kernel transpose"
      << " with alpha does not match at absolute index: "
      << result[1] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";

  ASSERT_EQ(-1, result[2]) << "Result of 3x1 macro full kernel transpose"
      << " with beta does not match at absolute index: "
      << result[2] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";

  ASSERT_EQ(-1, result[3]) << "Result of 3x1 macro full kernel transpose"
      << " with both coefficients does not match at absolute index: "
      << result[3] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";
}


TYPED_TEST(TestMacroTransVec, TestMacroFull3x2) {
  typename TestMacroTransVec<TypeParam>::CaseGenerator<
      KernelTypeTrans::KERNEL_FULL, 3, 2> test_full(*this);
  auto result = test_full();
  ASSERT_EQ(-1, result[0]) << "Result of 3x2 macro full kernel transpose"
      << " without coefficients does not match at absolute index: "
      << result[0] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";

  ASSERT_EQ(-1, result[1]) << "Result of 3x2 macro full kernel transpose"
      << " with alpha does not match at absolute index: "
      << result[1] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";

  ASSERT_EQ(-1, result[2]) << "Result of 3x2 macro full kernel transpose"
      << " with beta does not match at absolute index: "
      << result[2] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";

  ASSERT_EQ(-1, result[3]) << "Result of 3x2 macro full kernel transpose"
      << " with both coefficients does not match at absolute index: "
      << result[3] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";
}


TYPED_TEST(TestMacroTransVec, TestMacroFull3x3) {
  typename TestMacroTransVec<TypeParam>::CaseGenerator<
      KernelTypeTrans::KERNEL_FULL, 3, 3> test_full(*this);
  auto result = test_full();
  ASSERT_EQ(-1, result[0]) << "Result of 3x3 macro full kernel transpose"
      << " without coefficients does not match at absolute index: "
      << result[0] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";

  ASSERT_EQ(-1, result[1]) << "Result of 3x3 macro full kernel transpose"
      << " with alpha does not match at absolute index: "
      << result[1] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";

  ASSERT_EQ(-1, result[2]) << "Result of 3x3 macro full kernel transpose"
      << " with beta does not match at absolute index: "
      << result[2] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";

  ASSERT_EQ(-1, result[3]) << "Result of 3x3 macro full kernel transpose"
      << " with both coefficients does not match at absolute index: "
      << result[3] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";
}


TYPED_TEST(TestMacroTransVec, TestMacroHalf1x1) {
  typename TestMacroTransVec<TypeParam>::CaseGenerator<
      KernelTypeTrans::KERNEL_HALF, 1, 1> test_half(*this);
  auto result = test_half();
  ASSERT_EQ(-1, result[0]) << "Result of 1x1 macro half kernel transpose"
      << " without coefficients does not match at absolute index: "
      << result[0] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";

  ASSERT_EQ(-1, result[1]) << "Result of 1x1 macro half kernel transpose"
      << " with alpha does not match at absolute index: "
      << result[1] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";

  ASSERT_EQ(-1, result[2]) << "Result of 1x1 macro half kernel transpose"
      << " with beta does not match at absolute index: "
      << result[2] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";

  ASSERT_EQ(-1, result[3]) << "Result of 1x1 macro half kernel transpose"
      << " with both coefficients does not match at absolute index: "
      << result[3] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";
}


TYPED_TEST(TestMacroTransVec, TestMacroHalf1x2) {
  typename TestMacroTransVec<TypeParam>::CaseGenerator<
      KernelTypeTrans::KERNEL_HALF, 1, 2> test_half(*this);
  auto result = test_half();
  ASSERT_EQ(-1, result[0]) << "Result of 1x2 macro half kernel transpose"
      << " without coefficients does not match at absolute index: "
      << result[0] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";

  ASSERT_EQ(-1, result[1]) << "Result of 1x2 macro half kernel transpose"
      << " with alpha does not match at absolute index: "
      << result[1] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";

  ASSERT_EQ(-1, result[2]) << "Result of 1x2 macro half kernel transpose"
      << " with beta does not match at absolute index: "
      << result[2] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";

  ASSERT_EQ(-1, result[3]) << "Result of 1x2 macro half kernel transpose"
      << " with both coefficients does not match at absolute index: "
      << result[3] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";
}


TYPED_TEST(TestMacroTransVec, TestMacroHalf1x3) {
  typename TestMacroTransVec<TypeParam>::CaseGenerator<
      KernelTypeTrans::KERNEL_HALF, 1, 3> test_half(*this);
  auto result = test_half();
  ASSERT_EQ(-1, result[0]) << "Result of 1x3 macro half kernel transpose"
      << " without coefficients does not match at absolute index: "
      << result[0] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";

  ASSERT_EQ(-1, result[1]) << "Result of 1x3 macro half kernel transpose"
      << " with alpha does not match at absolute index: "
      << result[1] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";

  ASSERT_EQ(-1, result[2]) << "Result of 1x3 macro half kernel transpose"
      << " with beta does not match at absolute index: "
      << result[2] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";

  ASSERT_EQ(-1, result[3]) << "Result of 1x3 macro half kernel transpose"
      << " with both coefficients does not match at absolute index: "
      << result[3] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";
}


TYPED_TEST(TestMacroTransVec, TestMacroHalf2x1) {
  typename TestMacroTransVec<TypeParam>::CaseGenerator<
      KernelTypeTrans::KERNEL_HALF, 2, 1> test_half(*this);
  auto result = test_half();
  ASSERT_EQ(-1, result[0]) << "Result of 2x1 macro half kernel transpose"
      << " without coefficients does not match at absolute index: "
      << result[0] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";

  ASSERT_EQ(-1, result[1]) << "Result of 2x1 macro half kernel transpose"
      << " with alpha does not match at absolute index: "
      << result[1] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";

  ASSERT_EQ(-1, result[2]) << "Result of 2x1 macro half kernel transpose"
      << " with beta does not match at absolute index: "
      << result[2] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";

  ASSERT_EQ(-1, result[3]) << "Result of 2x1 macro half kernel transpose"
      << " with both coefficients does not match at absolute index: "
      << result[3] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";
}


TYPED_TEST(TestMacroTransVec, TestMacroHalf2x3) {
  typename TestMacroTransVec<TypeParam>::CaseGenerator<
      KernelTypeTrans::KERNEL_HALF, 2, 3> test_half(*this);
  auto result = test_half();
  ASSERT_EQ(-1, result[0]) << "Result of 2x3 macro half kernel transpose"
      << " without coefficients does not match at absolute index: "
      << result[0] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";

  ASSERT_EQ(-1, result[1]) << "Result of 2x3 macro half kernel transpose"
      << " with alpha does not match at absolute index: "
      << result[1] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";

  ASSERT_EQ(-1, result[2]) << "Result of 2x3 macro half kernel transpose"
      << " with beta does not match at absolute index: "
      << result[2] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";

  ASSERT_EQ(-1, result[3]) << "Result of 2x3 macro half kernel transpose"
      << " with both coefficients does not match at absolute index: "
      << result[3] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";
}


TYPED_TEST(TestMacroTransVec, TestMacroHalf3x1) {
  typename TestMacroTransVec<TypeParam>::CaseGenerator<
      KernelTypeTrans::KERNEL_HALF, 3, 1> test_half(*this);
  auto result = test_half();
  ASSERT_EQ(-1, result[0]) << "Result of 3x1 macro half kernel transpose"
      << " without coefficients does not match at absolute index: "
      << result[0] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";

  ASSERT_EQ(-1, result[1]) << "Result of 3x1 macro half kernel transpose"
      << " with alpha does not match at absolute index: "
      << result[1] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";

  ASSERT_EQ(-1, result[2]) << "Result of 3x1 macro half kernel transpose"
      << " with beta does not match at absolute index: "
      << result[2] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";

  ASSERT_EQ(-1, result[3]) << "Result of 3x1 macro half kernel transpose"
      << " with both coefficients does not match at absolute index: "
      << result[3] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";
}


TYPED_TEST(TestMacroTransVec, TestMacroHalf3x2) {
  typename TestMacroTransVec<TypeParam>::CaseGenerator<
      KernelTypeTrans::KERNEL_HALF, 3, 2> test_half(*this);
  auto result = test_half();
  ASSERT_EQ(-1, result[0]) << "Result of 3x2 macro half kernel transpose"
      << " without coefficients does not match at absolute index: "
      << result[0] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";

  ASSERT_EQ(-1, result[1]) << "Result of 3x2 macro half kernel transpose"
      << " with alpha does not match at absolute index: "
      << result[1] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";

  ASSERT_EQ(-1, result[2]) << "Result of 3x2 macro half kernel transpose"
      << " with beta does not match at absolute index: "
      << result[2] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";

  ASSERT_EQ(-1, result[3]) << "Result of 3x2 macro half kernel transpose"
      << " with both coefficients does not match at absolute index: "
      << result[3] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";
}


TYPED_TEST(TestMacroTransVec, TestMacroHalf3x3) {
  typename TestMacroTransVec<TypeParam>::CaseGenerator<
      KernelTypeTrans::KERNEL_HALF, 3, 3> test_half(*this);
  auto result = test_half();
  ASSERT_EQ(-1, result[0]) << "Result of 3x3 macro half kernel transpose"
      << " without coefficients does not match at absolute index: "
      << result[0] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";

  ASSERT_EQ(-1, result[1]) << "Result of 3x3 macro half kernel transpose"
      << " with alpha does not match at absolute index: "
      << result[1] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";

  ASSERT_EQ(-1, result[2]) << "Result of 3x3 macro half kernel transpose"
      << " with beta does not match at absolute index: "
      << result[2] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";

  ASSERT_EQ(-1, result[3]) << "Result of 3x3 macro half kernel transpose"
      << " with both coefficients does not match at absolute index: "
      << result[3] << ", offsets are: Original: (" << result[4] << ", "
      << result[5] << "), Actual: (" << result[6] << ", " << result[7] << ").";
}


TYPED_TEST(TestMacroTransScalar, TestCoefNone) {
  typename TestMacroTransScalar<TypeParam>::CaseGenerator<
      CoefUsageTrans::USE_NONE> test_none(*this);
  TensorIdx result = test_none();
  ASSERT_EQ(-1, result) << "Result of scalar macro without coefficient is"
      << " incorrect: Origin: " << this->org_data << ", Reference: "
      << this->ref_data << ", Actual: " << this->act_data;
}


TYPED_TEST(TestMacroTransScalar, TestCoefAlpha) {
  typename TestMacroTransScalar<TypeParam>::CaseGenerator<
      CoefUsageTrans::USE_ALPHA> test_alpha(*this);
  TensorIdx result = test_alpha();
  ASSERT_EQ(-1, result) << "Result of scalar macro with alpha is incorrect: "
      << "Origin: " << this->org_data << ", Reference: " << this->ref_data
      << ", Actual: " << this->act_data << ", alpha: " << this->alpha;
}


TYPED_TEST(TestMacroTransScalar, TestCoefBeta) {
  typename TestMacroTransScalar<TypeParam>::CaseGenerator<
      CoefUsageTrans::USE_BETA> test_beta(*this);
  TensorIdx result = test_beta();
  ASSERT_EQ(-1, result) << "Result of scalar macro with beta is incorrect: "
      << "Origin: " << this->org_data << ", Reference: " << this->ref_data
      << ", Actual: " << this->act_data << ", beta: " << this->beta;
}


TYPED_TEST(TestMacroTransScalar, TestCoefBoth) {
  typename TestMacroTransScalar<TypeParam>::CaseGenerator<
      CoefUsageTrans::USE_BOTH> test_both(*this);
  TensorIdx result = test_both();
  ASSERT_EQ(-1, result) << "Result of scalar macro with both coefficient is"
      << " incorrect: Origin: " << this->org_data << ", Reference: "
      << this->ref_data << ", Actual: " << this->act_data << ", alpha: "
      << this->alpha << ", beta: " << this->beta;
}

#endif // HPTC_UNIT_TEST_KERNELS_TEST_MACRO_KERNEL_TRANS_H_
