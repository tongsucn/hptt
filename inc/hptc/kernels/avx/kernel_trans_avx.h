#pragma once
#ifndef HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_H_
#define HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_H_

#include <xmmintrin.h>
#include <immintrin.h>

#include <hptc/types.h>
#include <hptc/param/parameter_trans.h>
#include <hptc/kernels/kernel_trans_base.h>


namespace hptc {

template <typename FloatType,
          KernelType TYPE>
struct KernelTransAvxBase {
};


template <typename FloatType,
          CoefUsage USAGE,
          KernelType TYPE = KernelType::KERNEL_FULL>
struct KernelTransAvx final : public KernelTransAvxBase<FloatType, TYPE> {
};


/*
 * Import implementation
 */
#include "kernel_trans_avx.tcc"

}

#endif // HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_H_
