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
#include <hptc/kernels/kernel_trans.h>
#include <hptc/kernels/macro_kernel_trans.h>

using namespace std;
using namespace hptc;


template <typename FloatType>
class TestOperationTrans : public ::testing::Test {
protected:
  using Deduced = DeducedFloatType<FloatType>;

  template <TensorOrder ORDER>
  struct CaseGenerator {
  };


  using Full = KernelTransFull<FloatType, CoefUsage::USE_NONE>
  MacroTransVec<FloatType, Full, 4, 4> macro_vec_4x4;
  MacroTransVec<FloatType, Full, 1, 4> macro_vec_1x4;
  MacroTransVec<FloatType, Full, 4, 1> macro_vec_4x1;
  MacroTransVec<FloatType, Full, 1, 1> macro_vec_1x1;
  MacroTransScalar<FloatType, CoefUsage::USE_NONE> macro_scalar;
};
