#pragma once
#ifndef HPTC_COMPAT_H_
#define HPTC_COMPAT_H_

#if defined(__INTEL_COMPILER)
  #define INLINE __forceinline
#elif defined(__GNUC__) || defined(__GNUG__) || defined(__clang__)
  #define INLINE __attribute__((always_inline))
#else
  #define INLINE inline
#endif


#define RESTRICT __restrict__


namespace hptc {

enum class MemLayout : bool {
  COL_MAJOR = true,
  ROW_MAJOR = false
};

}

#endif // HPTC_COMPAT_H_
