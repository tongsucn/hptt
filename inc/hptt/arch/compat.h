#pragma once
#ifndef HPTT_COMPAT_H_
#define HPTT_COMPAT_H_


// Force inline
#if defined(__INTEL_COMPILER)
  #define HPTT_INL __forceinline
#else
  #define HPTT_INL inline
#endif


#define RESTRICT __restrict__


namespace hptt {

enum class MemLayout : bool {
  COL_MAJOR = true,
  ROW_MAJOR = false
};

}

#endif // HPTT_COMPAT_H_
