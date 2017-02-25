#pragma once
#ifndef HPTC_CONFIG_CONFIG_TRANS_H_
#define HPTC_CONFIG_CONFIG_TRANS_H_

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
  KERNEL_FLIN = 2,
  KERNEL_HLIN = 3
};


template <typename FloatType,
          KernelTypeTrans TYPE,
          typename Enable = void>
struct RegTypeDeducer {
};


template <typename FloatType,
          KernelTypeTrans TYPE>
using DeducedRegType = typename RegTypeDeducer<FloatType, TYPE>::type;

}

#endif // HPTC_CONFIG_CONFIG_TRANS_H_
