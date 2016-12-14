#pragma once
#ifndef HPTC_KERNELS_KERNEL_TRANS_H_
#define HPTC_KERNELS_KERNEL_TRANS_H_

#include <hptc/kernels/avx/kernel_trans_avx.h>


namespace hptc {

template <typename FloatType,
          CoefUsage USAGE>
using KernelTrans = KernelTransAvxImpl<FloatType, USAGE>;

}

#endif // HPTC_KERNELS_KERNEL_TRANS_H_
