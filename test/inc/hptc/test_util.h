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
#include <hptc/arch/compat.h>


namespace hptc {

template <typename FloatType>
class DataWrapper {
public:
  DataWrapper(const std::vector<TensorIdx> &size, bool randomize = false);
  ~DataWrapper();

  void reset_ref();
  void reset_act();
  void trash_cache();
  static TensorInt verify(const FloatType *ref_data, const FloatType *act_data,
      TensorIdx data_len);
  TensorInt verify();

  FloatType *org_in_data, *org_out_data, *ref_data, *act_data;

protected:
  using TrashType_ = double;
  using Deduced_ = DeducedFloatType<FloatType>;

  constexpr static Deduced_ ele_lower_ = static_cast<Deduced_>(-50.0f);
  constexpr static Deduced_ ele_upper_ = static_cast<Deduced_>(50.0f);
  constexpr static TensorUInt inner_ = sizeof(FloatType) / sizeof(Deduced_);
  constexpr static TensorUInt trash_size_ = (1 << 20) * 100;
  constexpr static TrashType_ trash_calc_scale_ = 0.42;

  std::mt19937 gen_;
  std::uniform_real_distribution<Deduced_> dist_;
  const TensorIdx data_len_, page_size_;

  TrashType_ *trash_[2];
};


/**
 * \author Paul Springer
 * \author Tong Su
 */
template <typename FloatType>
struct RefTrans {
  void operator()(const FloatType * RESTRICT data_in,
      FloatType * RESTRICT data_out, const std::vector<TensorIdx> &size,
      const std::vector<TensorUInt> &perm,
      const DeducedFloatType<FloatType> alpha,
      const DeducedFloatType<FloatType> beta);
};


/*
 * Import implementation and explicit instantiation declaration
 * for class DataWrapper
 */
#include "test_util.tcc"

}

#endif // HPTC_TEST_UTIL_H_
