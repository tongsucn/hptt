#pragma once
#ifndef HPTC_KERNELS_KERNEL_TRANS_BASE_H_
#define HPTC_KERNELS_KERNEL_TRANS_BASE_H_

#include <hptc/types.h>


namespace hptc {

enum class KernelType : bool {
  KERNEL_FULL = true,
  KERNEL_HALF = false
};

}

#endif // HPTC_KERNELS_KERNEL_TRANS_BASE_H_
