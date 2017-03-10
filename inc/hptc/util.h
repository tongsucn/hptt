#pragma once
#ifndef HPTC_UTIL_H_
#define HPTC_UTIL_H_

#include <array>
#include <chrono>
#include <random>
#include <numeric>
#include <functional>
#include <utility>
#include <algorithm>

#include <hptc/types.h>


namespace hptc {

template <GenNumType GEN_NUM>
struct GenCounter {
};


template <GenNumType ROWS,
          GenNumType COLS>
struct DualCounter {
};


template <typename FloatType>
class DataWrapper {
public:
  DataWrapper(const std::vector<TensorOrder> &size, bool randomize = true);
  DataWrapper(const std::vector<TensorOrder> &size, const FloatType *input_data,
      const FloatType *output_data);
  virtual ~DataWrapper();

  FloatType *org_in_data, *org_out_data;

protected:
  using Deduced_ = DeducedFloatType<FloatType>;

  constexpr static Deduced_ ele_lower_ = static_cast<Deduced_>(-50.0f);
  constexpr static Deduced_ ele_upper_ = static_cast<Deduced_>(50.0f);
  constexpr static GenNumType inner_ = sizeof(FloatType) / sizeof(Deduced_);

  std::mt19937 gen_;
  std::uniform_real_distribution<Deduced_> dist_;
  TensorIdx data_len_;
};


class TimerWrapper {
public:
  TimerWrapper(GenNumType times) : times_(times) {}

  template <typename Callable,
            typename... Args>
  INLINE double operator()(Callable &target, Args&&... args);

private:
  GenNumType times_;
};


/*
 * Import implementation
 */
#include "util.tcc"

}

#endif // HPTC_UTIL_H_
