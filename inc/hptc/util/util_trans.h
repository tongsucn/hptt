#pragma once
#ifndef HPTC_UTIL_UTIL_TRANS_H_
#define HPTC_UTIL_UTIL_TRANS_H_

#include <array>
#include <vector>
#include <algorithm>

#include <hptc/types.h>


namespace hptc {

/*
 * Transpose kernel types
 */
enum class KernelTypeTrans : TensorUInt {
  KERNEL_FULL = 0,
  KERNEL_HALF = 1,
  KERNEL_LINE = 2
};


template <TensorUInt ORDER>
struct LoopParamTrans {
  LoopParamTrans();

  void set_pass(TensorUInt order);
  void set_disable();
  bool is_disabled() const;

  TensorIdx loop_begin[ORDER];
  TensorIdx loop_end[ORDER];
  TensorIdx loop_step[ORDER];
};


template <TensorUInt ORDER>
using LoopOrderTrans = std::array<TensorUInt, ORDER>;


template <TensorUInt ORDER>
using ParaStrategyTrans = std::array<TensorUInt, ORDER>;


template <typename FloatType>
double calc_tp_trans(const std::vector<TensorIdx> &size, double time_ms);


/*
 * Import implementation
 */
#include "util_trans.tcc"

}

#endif // HPTC_UTIL_UTIL_TRANS_H_
