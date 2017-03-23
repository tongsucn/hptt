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
  TimerWrapper(TensorUInt times);

  template <typename Callable,
            typename... Args>
  INLINE double operator()(Callable &target, Args&&... args);

private:
  TensorUInt times_;
};


template <typename ValType>
struct ModCmp {
  INLINE bool operator()(const ValType &first, const ValType &second);
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
