#pragma once
#ifndef HPTC_KERNELS_MICRO_KERNEL_TRANS_H_
#define HPTC_KERNELS_MICRO_KERNEL_TRANS_H_

#if defined HPTC_ARCH_AVX2
// AVX2
#include <hptc/arch/avx2/kernel_trans_avx2.h>

#elif defined HPTC_ARCH_AVX
// AVX
#include <hptc/arch/avx/kernel_trans_avx.h>

#elif defined HPTC_ARCH_IBM
// PowerPC
#include <hptc/arch/ibm/kernel_trans_ibm.h>

#else
// Common
#include <hptc/arch/common/kernel_trans_common.h>

#endif

#include <hptc/util/util_trans.h>


namespace hptc {

/*
 * Alias for micro kernels
 */
template <typename FloatType>
using KernelTransFull = KernelTrans<FloatType, KernelTypeTrans::KERNEL_FULL>;

template <typename FloatType>
using KernelTransHalf = KernelTrans<FloatType, KernelTypeTrans::KERNEL_HALF>;

template <typename FloatType>
using KernelTransLinear = KernelTrans<FloatType, KernelTypeTrans::KERNEL_LINE>;

}

#endif // HPTC_KERNELS_MICRO_KERNEL_TRANS_H_
