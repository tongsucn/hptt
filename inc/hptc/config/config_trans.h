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


enum class KernelTypeTrans : bool {
  KERNEL_FULL = true,
  KERNEL_HALF = false
};

}

#endif // HPTC_CONFIG_CONFIG_TRANS_H_
