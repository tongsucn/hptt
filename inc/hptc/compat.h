#pragma once
#ifndef HPTC_COMPAT_H_
#define HPTC_COMPAT_H_

// Force inline
#if defined(__INTEL_COMPILER)
  #define HPTC_INL __forceinline
#elif defined(__GNUC__) || defined(__GNUG__) || defined(__clang__)
  #define HPTC_INL inline
#else
  #define HPTC_INL inline
#endif


#define RESTRICT __restrict__


namespace hptc {

enum class MemLayout : bool {
  COL_MAJOR = true,
  ROW_MAJOR = false
};

}

#endif // HPTC_COMPAT_H_
