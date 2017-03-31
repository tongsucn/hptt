#include <hptc/arch/common/kernel_trans_common.h>

#include <hptc/types.h>
#include <hptc/arch/compat.h>
#include <hptc/util/util_trans.h>


namespace hptc {

template <typename FloatType,
          KernelTypeTrans TYPE>
void KernelTrans<FloatType, TYPE>::set_coef(
    const DeducedFloatType<FloatType> alpha,
    const DeducedFloatType<FloatType> beta) {
  this->reg_alpha_ = alpha, this->reg_beta_ = beta;
}


template <typename FloatType,
          KernelTypeTrans TYPE>
void KernelTrans<FloatType, TYPE>::exec(const FloatType * RESTRICT in_data,
    FloatType * RESTRICT out_data, const TensorIdx input_stride,
    const TensorIdx output_stride) const {
  // Get number of elements to be processed in on row
  constexpr auto WIDTH = KernelTrans<FloatType, TYPE>::KN_WIDTH;

  for (hptc::TensorUInt ncont_idx = 0; ncont_idx < WIDTH; ++ncont_idx) {
    for (hptc::TensorUInt cont_idx = 0; cont_idx < WIDTH; ++cont_idx) {
      const auto input_idx = cont_idx + ncont_idx * input_stride,
            output_idx = ncont_idx + cont_idx * output_stride;
      out_data[output_idx] = this->reg_alpha_ * in_data[input_idx]
            + this->reg_beta_ * out_data[output_idx];
    }
  }
}


/*
 * Explicit template instantiation definition for class KernelTrans
 */
template class KernelTrans<float, KernelTypeTrans::KERNEL_FULL>;
template class KernelTrans<double, KernelTypeTrans::KERNEL_FULL>;
template class KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_FULL>;
template class KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_FULL>;

template class KernelTrans<float, KernelTypeTrans::KERNEL_HALF>;
template class KernelTrans<double, KernelTypeTrans::KERNEL_HALF>;
template class KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_HALF>;
template class KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_HALF>;

template class KernelTrans<float, KernelTypeTrans::KERNEL_LINE>;
template class KernelTrans<double, KernelTypeTrans::KERNEL_LINE>;
template class KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_LINE>;
template class KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_LINE>;

}
