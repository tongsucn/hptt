#pragma once
#ifndef HPTC_CONFIG_CONFIG_TRANS_H_
#define HPTC_CONFIG_CONFIG_TRANS_H_

#include <array>

#include <hptc/types.h>


namespace hptc {

enum class CoefUsageTrans : GenNumType {
  USE_NONE  = 0x0,
  USE_ALPHA = 0x1,
  USE_BETA  = 0x2,
  USE_BOTH  = 0x3
};


enum class KernelTypeTrans : GenNumType {
  KERNEL_FULL = 0,
  KERNEL_HALF = 1,
  KERNEL_LINE = 2
};


template <typename FloatType,
          KernelTypeTrans TYPE,
          typename Enable = void>
struct RegTypeDeducer {
};


template <typename FloatType,
          KernelTypeTrans TYPE>
using DeducedRegType = typename RegTypeDeducer<FloatType, TYPE>::type;


template <TensorOrder ORDER>
struct LoopParamTrans {
  LoopParamTrans();

  INLINE void set_pass(TensorOrder order);
  INLINE void set_disable();
  INLINE bool is_disabled();

  TensorIdx loop_begin[ORDER];
  TensorIdx loop_end[ORDER];
  TensorIdx loop_step[ORDER];
};


template <TensorOrder ORDER>
using LoopOrderTrans = std::array<TensorOrder, ORDER>;


template <TensorOrder ORDER>
using ParaStrategyTrans = std::array<GenNumType, ORDER>;


template <TensorOrder ORDER>
using LoopGroupTrans = std::array<LoopParamTrans<ORDER>, 9>;


/*
 * Import implementation
 */
#include "config_trans.tcc"

}

#endif // HPTC_CONFIG_CONFIG_TRANS_H_
