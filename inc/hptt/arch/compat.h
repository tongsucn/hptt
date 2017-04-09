#pragma once
#ifndef HPTT_COMPAT_H_
#define HPTT_COMPAT_H_

#include <cstdint>


#if defined(__INTEL_COMPILER)
  #define HPTT_INL __forceinline
  #define HPTT_MEM_ALIGN __attribute__((aligned(64)))
#elif defined(__GNUC__) || defined(__GNUG__) || defined(__clang__)
  #define HPTT_INL inline
  #define HPTT_MEM_ALIGN __attribute__((aligned(64)))
#else
  #define HPTT_INL inline
  #define HPTT_MEM_ALIGN
#endif


#define RESTRICT __restrict__


namespace hptt {

template <typename FloatType>
bool check_aligned(const FloatType *begin_ptr) {
  return reinterpret_cast<intptr_t>(begin_ptr) % 32 == 0;
}

}

#endif // HPTT_COMPAT_H_
