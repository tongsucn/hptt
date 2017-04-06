#pragma once
#ifndef HPTT_KERNELS_MICRO_KERNEL_TRANS_H_
#define HPTT_KERNELS_MICRO_KERNEL_TRANS_H_

#if defined HPTT_ARCH_AVX2
// AVX2
#include <hptt/arch/avx2/kernel_trans_avx2.h>

#elif defined HPTT_ARCH_AVX
// AVX
#include <hptt/arch/avx/kernel_trans_avx.h>

#elif defined HPTT_ARCH_IBM
// PowerPC
#include <hptt/arch/ibm/kernel_trans_ibm.h>

#else
// Common
#include <hptt/arch/common/kernel_trans_common.h>

#endif

#include <hptt/util/util_trans.h>


namespace hptt {

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

#endif // HPTT_KERNELS_MICRO_KERNEL_TRANS_H_
