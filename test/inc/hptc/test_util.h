#pragma once
#ifndef HPTC_TEST_UTIL_H_
#define HPTC_TEST_UTIL_H_

#include <cstdlib>

#include <vector>
#include <random>
#include <numeric>
#include <functional>

#include <unistd.h>

#include <hptc/types.h>


namespace hptc {

template <typename FloatType>
class DataWrapper {
public:
  DataWrapper(const std::vector<TensorOrder> &size, bool randomize = false);
  ~DataWrapper();

  void reset_ref();
  void reset_act();
  static TensorIdx verify(const FloatType *ref_data, const FloatType *act_data,
      TensorIdx data_len);
  TensorIdx verify();

  FloatType *org_in_data, *org_out_data, *ref_data, *act_data;

protected:
  using Deduced_ = DeducedFloatType<FloatType>;

  constexpr static Deduced_ ele_lower_ = static_cast<Deduced_>(-50.0f);
  constexpr static Deduced_ ele_upper_ = static_cast<Deduced_>(50.0f);
  constexpr static GenNumType inner_ = sizeof(FloatType) / sizeof(Deduced_);
  constexpr static GenNumType trash_size_ = sizeof(FloatType) * (1 << 20) * 100;

  std::mt19937 gen_;
  std::uniform_real_distribution<Deduced_> dist_;
  const TensorIdx data_len_, page_size_;

  FloatType *trash_[2];
};


/*
 * Import implementation and explicit instantiation declaration
 * for class DataWrapper
 */
#include "test_util.tcc"

}

#endif // HPTC_TEST_UTIL_H_
