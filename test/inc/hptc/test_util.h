#pragma once
#ifndef TEST_TEST_UTIL_H_
#define TEST_TEST_UTIL_H_

#include <cmath>

#include <vector>
#include <random>
#include <numeric>
#include <functional>
#include <algorithm>

#include <hptc/util.h>
#include <hptc/types.h>
#include <hptc/compat.h>


namespace hptc {

template <typename FloatType>
class TestDataWrapper final : public DataWrapper<FloatType> {
public:
  TestDataWrapper(const std::vector<TensorOrder> &size);
  ~TestDataWrapper();

  void reset_ref();
  void reset_act();
  static TensorIdx verify(const FloatType *ref_data, const FloatType *act_data,
      TensorIdx data_len);
  TensorIdx verify();

  FloatType *ref_data, *act_data;
};


/*
 * Import implementation.
 */
#include "test_util.tcc"

}

#endif // TEST_TEST_UTIL_H_
