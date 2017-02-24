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


template <TensorOrder ORDER>
using LoopOrder = std::array<TensorOrder, ORDER>;


template <TensorOrder ORDER>
struct LoopParam {
  LoopParam();

  INLINE void set_pass(TensorOrder order);
  INLINE void set_disable();
  INLINE bool is_disabled();

  TensorIdx loop_begin[ORDER];
  TensorIdx loop_end[ORDER];
  TensorIdx loop_step[ORDER];
};


template <typename FloatType>
class DataWrapper {
public:
  DataWrapper(const std::vector<TensorOrder> &size);
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
