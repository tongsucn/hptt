#pragma once
#ifndef HPTC_KERNELS_COMMON_KERNEL_TRANS_COMMON_H_
#define HPTC_KERNELS_COMMON_KERNEL_TRANS_COMMON_H_

#include <xmmintrin.h>
#include <immintrin.h>

#include <type_traits>

#include <hptc/types.h>
#include <hptc/util/util.h>
#include <hptc/util/util_trans.h>


namespace hptc {

#define REG_SIZE_BYTE_COMMON 32


template <typename FloatType,
          KernelTypeTrans TYPE>
struct RegTypeDeducer<FloatType, TYPE, void> {
  using type = DeducedFloatType<FloatType>;
};


template <typename FloatType,
          KernelTypeTrans TYPE>
struct KernelTransCommonBase {
  using Float = FloatType;
  using RegType = DeducedRegType<FloatType, TYPE>;

  static constexpr TensorUInt kn_width = KernelTypeTrans::KERNEL_FULL == TYPE
      ? REG_SIZE_BYTE_COMMON / sizeof(FloatType)
      : KernelTypeTrans::KERNEL_HALF == TYPE
          ? REG_SIZE_BYTE_COMMON / sizeof(FloatType) / 2 : 1;
};


template <typename FloatType,
          CoefUsageTrans USAGE,
          KernelTypeTrans TYPE>
struct KernelTransCommon final : public KernelTransCommonBase<FloatType, TYPE> {
  using RegType = DeducedRegType<FloatType, TYPE>;

  static RegType reg_coef(const DeducedFloatType<FloatType> coef);

  void operator()(const FloatType * RESTRICT input_data,
      FloatType * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride, const RegType &reg_alpha,
      const RegType &reg_beta) const;
};


/*
 * Type alias for common micro kernel
 */
template <typename FloatType,
          CoefUsageTrans USAGE,
          KernelTypeTrans TYPE>
using KernelTrans = KernelTransCommon<FloatType, USAGE, TYPE>;


/*
 * Import template class KernelTransCommon's partial specialization
 * and explicit instantiation declaration
 */
#include "kernel_trans_common.tcc"

}

#endif // HPTC_KERNELS_COMMON_KERNEL_TRANS_COMMON_H_
