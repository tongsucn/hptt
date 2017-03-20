#pragma once
#ifndef HPTC_UTIL_H_
#define HPTC_UTIL_H_

#include <vector>
#include <random>
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

#endif // HPTC_UTIL_H_
