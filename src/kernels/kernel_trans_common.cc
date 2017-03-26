#include <hptc/kernels/common/kernel_trans_common.h>

#include <xmmintrin.h>
#include <immintrin.h>

#include <hptc/types.h>
#include <hptc/compat.h>
#include <hptc/util/util_trans.h>


namespace hptc {

/*
 * Implementation for class KernelTransCommon
 */
template <typename FloatType,
          CoefUsageTrans USAGE,
          KernelTypeTrans TYPE>
HPTC_INL DeducedRegType<FloatType, TYPE>
KernelTransCommon<FloatType, USAGE, TYPE>:: reg_coef(
    const DeducedFloatType<FloatType> coef) {
  return coef;
}


template <typename FloatType,
          CoefUsageTrans USAGE,
          KernelTypeTrans TYPE>
HPTC_INL void KernelTransCommon<FloatType, USAGE, TYPE>::exec(
    const FloatType * RESTRICT input_data, FloatType * RESTRICT output_data,
    const TensorIdx input_stride, const TensorIdx output_stride,
    const RegType &reg_alpha, const RegType &reg_beta) {
  // Get number of elements to be processed in on row
  constexpr auto KN_WIDTH = KernelTransCommonBase<FloatType, TYPE>::kn_width;

  for (TensorUInt ncont_idx = 0; ncont_idx < KN_WIDTH; ++ncont_idx) {
    for (TensorUInt cont_idx = 0; cont_idx < KN_WIDTH; ++cont_idx) {
      const auto input_idx = cont_idx + ncont_idx * input_stride,
            output_idx = ncont_idx + cont_idx * output_stride;
      if (CoefUsageTrans::USE_BOTH == USAGE)
        output_data[output_idx] = reg_alpha * input_data[input_idx]
            + reg_beta * output_data[output_idx];
      else if (CoefUsageTrans::USE_ALPHA == USAGE)
        output_data[output_idx] = reg_alpha * input_data[input_idx];
      else if (CoefUsageTrans::USE_BETA == USAGE)
        output_data[output_idx] = input_data[input_idx]
            + reg_beta * output_data[output_idx];
      else
        output_data[output_idx] = input_data[input_idx];
    }
  }
}


/*
 * Explicit instantiation for struct KernelTransCommon
 */
template struct KernelTransCommon<float, CoefUsageTrans::USE_NONE,
         KernelTypeTrans::KERNEL_FULL>;
template struct KernelTransCommon<float, CoefUsageTrans::USE_ALPHA,
         KernelTypeTrans::KERNEL_FULL>;
template struct KernelTransCommon<float, CoefUsageTrans::USE_BETA,
         KernelTypeTrans::KERNEL_FULL>;
template struct KernelTransCommon<float, CoefUsageTrans::USE_BOTH,
         KernelTypeTrans::KERNEL_FULL>;
template struct KernelTransCommon<double, CoefUsageTrans::USE_NONE,
         KernelTypeTrans::KERNEL_FULL>;
template struct KernelTransCommon<double, CoefUsageTrans::USE_ALPHA,
         KernelTypeTrans::KERNEL_FULL>;
template struct KernelTransCommon<double, CoefUsageTrans::USE_BETA,
         KernelTypeTrans::KERNEL_FULL>;
template struct KernelTransCommon<double, CoefUsageTrans::USE_BOTH,
         KernelTypeTrans::KERNEL_FULL>;
template struct KernelTransCommon<FloatComplex, CoefUsageTrans::USE_NONE,
         KernelTypeTrans::KERNEL_FULL>;
template struct KernelTransCommon<FloatComplex, CoefUsageTrans::USE_ALPHA,
         KernelTypeTrans::KERNEL_FULL>;
template struct KernelTransCommon<FloatComplex, CoefUsageTrans::USE_BETA,
         KernelTypeTrans::KERNEL_FULL>;
template struct KernelTransCommon<FloatComplex, CoefUsageTrans::USE_BOTH,
         KernelTypeTrans::KERNEL_FULL>;
template struct KernelTransCommon<DoubleComplex, CoefUsageTrans::USE_NONE,
         KernelTypeTrans::KERNEL_FULL>;
template struct KernelTransCommon<DoubleComplex, CoefUsageTrans::USE_ALPHA,
         KernelTypeTrans::KERNEL_FULL>;
template struct KernelTransCommon<DoubleComplex, CoefUsageTrans::USE_BETA,
         KernelTypeTrans::KERNEL_FULL>;
template struct KernelTransCommon<DoubleComplex, CoefUsageTrans::USE_BOTH,
         KernelTypeTrans::KERNEL_FULL>;
template struct KernelTransCommon<float, CoefUsageTrans::USE_NONE,
         KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransCommon<float, CoefUsageTrans::USE_ALPHA,
         KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransCommon<float, CoefUsageTrans::USE_BETA,
         KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransCommon<float, CoefUsageTrans::USE_BOTH,
         KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransCommon<double, CoefUsageTrans::USE_NONE,
         KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransCommon<double, CoefUsageTrans::USE_ALPHA,
         KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransCommon<double, CoefUsageTrans::USE_BETA,
         KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransCommon<double, CoefUsageTrans::USE_BOTH,
         KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransCommon<FloatComplex, CoefUsageTrans::USE_NONE,
         KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransCommon<FloatComplex, CoefUsageTrans::USE_ALPHA,
         KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransCommon<FloatComplex, CoefUsageTrans::USE_BETA,
         KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransCommon<FloatComplex, CoefUsageTrans::USE_BOTH,
         KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransCommon<DoubleComplex, CoefUsageTrans::USE_NONE,
         KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransCommon<DoubleComplex, CoefUsageTrans::USE_ALPHA,
         KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransCommon<DoubleComplex, CoefUsageTrans::USE_BETA,
         KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransCommon<DoubleComplex, CoefUsageTrans::USE_BOTH,
         KernelTypeTrans::KERNEL_HALF>;

}
