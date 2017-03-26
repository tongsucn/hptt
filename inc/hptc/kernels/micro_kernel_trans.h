#pragma once
#ifndef HPTC_KERNELS_MICRO_KERNEL_TRANS_H_
#define HPTC_KERNELS_MICRO_KERNEL_TRANS_H_

#include <hptc/util/util_trans.h>

// Architecture selection
#if defined(HPTC_ARCH_AVX)
// AVX
#include <hptc/kernels/avx/kernel_trans_avx.h>

#elif defined(HPTC_ARCH_AVX2)
// AVX2
#include <hptc/kernels/avx2/kernel_trans_avx2.h>

#else
// Common, no specific architecture
#include <hptc/kernels/common/kernel_trans_common.h>

#endif


namespace hptc {

/*
 * Alias for micro kernels
 */
template <typename FloatType,
          CoefUsageTrans USAGE>
using KernelTransFull = KernelTrans<FloatType, USAGE,
    KernelTypeTrans::KERNEL_FULL>;


template <typename FloatType,
          CoefUsageTrans USAGE>
using KernelTransHalf = KernelTrans<FloatType, USAGE,
    KernelTypeTrans::KERNEL_HALF>;

}

#endif // HPTC_KERNELS_MICRO_KERNEL_TRANS_H_
