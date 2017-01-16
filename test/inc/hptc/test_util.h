#pragma once
#ifndef TEST_TEST_UTIL_H_
#define TEST_TEST_UTIL_H_

#include <cmath>

#include <array>
#include <random>
#include <numeric>
#include <functional>
#include <algorithm>

#include <hptc/types.h>


namespace hptc {

template <typename FloatType>
struct DataWrapper {
  using Deduced = DeducedFloatType<FloatType>;

  template <GenNumType ORDER>
  DataWrapper(const std::array<TensorIdx, ORDER> &size);

  void init();
  void reset_ref();
  void reset_act();
  static TensorIdx verify(const FloatType *ref_data, const FloatType *act_data,
      TensorIdx data_len);
  TensorIdx verify();


  constexpr static Deduced ele_lower = static_cast<Deduced>(-500.0f);
  constexpr static Deduced ele_upper = static_cast<Deduced>(500.0f);
  constexpr static GenNumType inner = sizeof(FloatType) / sizeof(Deduced);

  std::mt19937 gen;
  std::uniform_real_distribution<Deduced> dist;
  TensorIdx data_len;
  FloatType *org_in_data, *org_out_data, *ref_data, *act_data;
};


/*
 * Import implementation.
 */
#include "test_util.tcc"

}

#endif // TEST_TEST_UTIL_H_
