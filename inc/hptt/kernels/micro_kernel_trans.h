#pragma once
#ifndef HPTT_KERNELS_MICRO_KERNEL_TRANS_H_
#define HPTT_KERNELS_MICRO_KERNEL_TRANS_H_

#if defined HPTT_ARCH_AVX2
// AVX2
#include <hptt/arch/avx2/kernel_trans_avx2.h>

#elif defined HPTT_ARCH_AVX
// AVX
#include <hptt/arch/avx/kernel_trans_avx.h>

#elif defined HPTT_ARCH_ARM
// ARM
#include <hptt/arch/arm/kernel_trans_arm.h>

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
template <typename FloatType,
          bool UPDATE_OUT>
using KernelTransFull = KernelTrans<FloatType, KernelTypeTrans::KERNEL_FULL,
    UPDATE_OUT>;

template <typename FloatType,
          bool UPDATE_OUT>
using KernelTransHalf = KernelTrans<FloatType, KernelTypeTrans::KERNEL_HALF,
    UPDATE_OUT>;

template <typename FloatType,
          bool UPDATE_OUT>
using KernelTransLinear = KernelTrans<FloatType, KernelTypeTrans::KERNEL_LINE,
    UPDATE_OUT>;

}

#endif // HPTT_KERNELS_MICRO_KERNEL_TRANS_H_
