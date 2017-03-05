#pragma once
#ifndef HPTC_KERNELS_KERNEL_TRANS_H_
#define HPTC_KERNELS_KERNEL_TRANS_H_

#include <hptc/types.h>
#include <hptc/config/config_trans.h>
#include <hptc/kernels/avx/kernel_trans_avx.h>


namespace hptc {

template <typename FloatType,
          CoefUsageTrans USAGE,
          KernelTypeTrans TYPE>
using KernelTrans = KernelTransAvx<FloatType, USAGE, TYPE>;


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


template <typename FloatType,
          CoefUsageTrans USAGE>
using KernelTransLinear = KernelTrans<FloatType, USAGE,
    KernelTypeTrans::KERNEL_LINE>;

}

#endif // HPTC_KERNELS_KERNEL_TRANS_H_
