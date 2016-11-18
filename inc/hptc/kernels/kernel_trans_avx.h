#pragma once
#ifndef HPTC_KERNEL_KERNEL_TRANS_AVX_H_
#define HPTC_KERNEL_KERNEL_TRANS_AVX_H_

#include <cstdint>
#include <xmmintrin.h>
#include <immintrin.h>

#include <memory>
#include <utility>

#include <hptc/param/parameter_trans.h>
#include <hptc/types.h>


namespace hptc {

template <typename FloatType,
          uint32_t HEIGHT = 0,
          uint32_t WIDTH = HEIGHT>
inline void kernel_trans(std::shared_ptr<ParamTrans<FloatType>> &param);


/*
 * Import implementation
 */
#include "kernel_trans_avx.tcc"

}

#endif // HPTC_KERNEL_KERNEL_TRANS_AVX_H_
