#pragma once
#ifndef HPTC_UTIL_H_
#define HPTC_UTIL_H_

#include <array>
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


/*
 * Import implementation
 */
#include "util.tcc"

}

#endif // HPTC_UTIL_H_
