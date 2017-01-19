#pragma once
#ifndef TEST_TEST_UTIL_H_
#define TEST_TEST_UTIL_H_

#include <cmath>

#include <chrono>
#include <vector>
#include <random>
#include <utility>
#include <numeric>
#include <functional>
#include <algorithm>

#include <hptc/types.h>
#include <hptc/compat.h>


namespace hptc {

template <typename FloatType>
struct DataWrapper {
  using Deduced = DeducedFloatType<FloatType>;

  DataWrapper(const std::vector<TensorOrder> &size);
  ~DataWrapper();

  void init();
  void reset_ref();
  void reset_act();
  static TensorIdx verify(const FloatType *ref_data, const FloatType *act_data,
      TensorIdx data_len);
  TensorIdx verify();


  constexpr static Deduced ele_lower = static_cast<Deduced>(-500.0f);
  constexpr static Deduced ele_upper = static_cast<Deduced>(500.0f);
  constexpr static GenNumType inner = sizeof(FloatType) / sizeof(Deduced);

  std::random_device rd;
  std::mt19937 gen;
  std::uniform_real_distribution<Deduced> dist;
  std::vector<TensorOrder> size;
  TensorIdx data_len;
  FloatType *org_in_data, *org_out_data, *ref_data, *act_data;
};


struct TimerWrapper {
  using Duration = std::chrono::duration<double, std::milli>;

  TimerWrapper(GenNumType times = 10);

  template <typename Callable,
            typename... Args>
  INLINE double operator()(Callable &target, Args&&... args);

  GenNumType times;
};


struct RefTransConfig {
  RefTransConfig(TensorOrder order, TensorOrder thread_num,
      const std::vector<TensorOrder> &perm,
      const std::vector<TensorOrder> &size);

  TensorOrder order;
  TensorOrder thread_num;
  std::vector<TensorOrder> perm;
  std::vector<TensorOrder> size;
};


/*
 * Import implementation.
 */
#include "test_util.tcc"

}

#endif // TEST_TEST_UTIL_H_
