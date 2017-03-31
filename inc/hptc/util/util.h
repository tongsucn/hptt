#pragma once
#ifndef HPTC_UTIL_UTIL_H_
#define HPTC_UTIL_UTIL_H_

#include <cfloat>

#include <vector>
#include <chrono>
#include <utility>
#include <type_traits>
#include <unordered_map>

#include <hptc/types.h>


namespace hptc {

template <TensorUInt GEN_NUM>
struct GenCounter {
};


template <TensorUInt ROWS,
          TensorUInt COLS>
struct DualCounter {
};


template <bool COND,
          typename Type = void>
using Enable = typename std::enable_if<COND, Type>::type;


class TimerWrapper {
public:
  TimerWrapper(const TensorUInt times);

  template <typename Callable,
            typename... Args>
  double operator()(Callable &target, Args&&... args);

  void start_countdown(const double timeout);
  bool is_timeout() const;

private:
  using Duration_ = std::chrono::duration<double, std::milli>;

  const TensorUInt times_;
  double timeout_;
  std::chrono::time_point<std::chrono::high_resolution_clock> countdown_begin_;
};


template <typename ValType>
struct ModCmp {
  bool operator()(const ValType &first, const ValType &second);
};


std::vector<TensorUInt> approx_prod(const std::vector<TensorUInt> &integers,
    const TensorUInt target);


std::unordered_map<TensorUInt, TensorUInt> factorize(TensorUInt target);


template <typename TargetFunc>
std::vector<TensorUInt> assign_factor(
    std::unordered_map<TensorUInt, TensorUInt> &fact_map, TensorUInt &target,
    TensorUInt &accumulate, TargetFunc cmp);


std::vector<TensorUInt> flat_map(
    const std::unordered_map<TensorUInt, TensorUInt> &input_map);


/*
 * Import implementation
 */
#include "util.tcc"

}

#endif // HPTC_UTIL_UTIL_H_
