#pragma once
#ifndef HPTC_COMPAT_H_
#define HPTC_COMPAT_H_

#if defined(__ICC) || defined(__INTEL_COMPILER)
  #define INLINE __forceinline
#elif defined(__GNUC__) || defined(__GNUG__)
  #define INLINE __attribute__((always_inline))
#else
  #define INLINE inline
#endif


#define RESTRICT __restrict__

#endif // HPTC_COMPAT_H_
