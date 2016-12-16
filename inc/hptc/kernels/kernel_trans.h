#pragma once
#ifndef HPTC_KERNELS_KERNEL_TRANS_H_
#define HPTC_KERNELS_KERNEL_TRANS_H_

#include <hptc/kernels/avx/kernel_trans_avx.h>


namespace hptc {

template <typename FloatType,
          CoefUsage USAGE,
          KernelTransType KERNEL_TYPE = KernelTransType::KERNEL_FULL>
using KernelTrans = KernelTransAvx<FloatType, USAGE, KERNEL_TYPE>;

}

#endif // HPTC_KERNELS_KERNEL_TRANS_H_
