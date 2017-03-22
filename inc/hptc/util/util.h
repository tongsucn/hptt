#pragma once
#ifndef HPTC_UTIL_UTIL_H_
#define HPTC_UTIL_UTIL_H_

#include <vector>
#include <chrono>
#include <utility>
#include <type_traits>
#include <unordered_map>

#include <hptc/types.h>


namespace hptc {

template <GenNumType GEN_NUM>
struct GenCounter {
};


template <GenNumType ROWS,
          GenNumType COLS>
struct DualCounter {
};


template <bool COND,
          typename Type = void>
using Enable = typename std::enable_if<COND, Type>::type;


class TimerWrapper {
public:
  TimerWrapper(GenNumType times);

  template <typename Callable,
            typename... Args>
  INLINE double operator()(Callable &target, Args&&... args);

private:
  GenNumType times_;
};


template <typename ValType>
struct ModCmp {
  INLINE bool operator()(const ValType &first, const ValType &second);
};


std::vector<GenNumType> approx_prod(const std::vector<GenNumType> &integers,
    const GenNumType target);


std::unordered_map<GenNumType, GenNumType> factorize(GenNumType target);


template <typename TargetFunc>
std::vector<GenNumType> assign_factor(
    std::unordered_map<GenNumType, GenNumType> &fact_map, GenNumType &target,
    GenNumType &accumulate, TargetFunc cmp);


std::vector<GenNumType> flat_map(
    const std::unordered_map<GenNumType, GenNumType> &input_map);


/*
 * Import implementation
 */
#include "util.tcc"

}

#endif // HPTC_UTIL_UTIL_H_
