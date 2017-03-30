#pragma once
#ifndef HPTC_COMPAT_H_
#define HPTC_COMPAT_H_

#include <cstdint>


// Force inline
#if defined(__INTEL_COMPILER)
  #define HPTC_INL __forceinline
#else
  #define HPTC_INL inline
#endif


#define RESTRICT __restrict__


namespace hptc {

enum class MemLayout : bool {
  COL_MAJOR = true,
  ROW_MAJOR = false
};


void hptc_cpuid(const uint32_t input, uint32_t output[4]);

}

#endif // HPTC_COMPAT_H_
