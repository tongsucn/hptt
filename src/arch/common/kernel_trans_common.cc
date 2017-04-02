#include <hptc/arch/common/kernel_trans_common.h>

#include <hptc/types.h>
#include <hptc/arch/compat.h>
#include <hptc/util/util_trans.h>


namespace hptc {

template <typename FloatType,
          KernelTypeTrans TYPE>
void KernelTransData<FloatType, TYPE>::set_coef(
    const DeducedFloatType<FloatType> alpha,
    const DeducedFloatType<FloatType> beta) {
  this->alpha_ = alpha, this->beta_ = beta;
}


template <typename FloatType,
          KernelTypeTrans TYPE>
void KernelTrans<FloatType, TYPE>::exec(const FloatType * RESTRICT in_data,
    FloatType * RESTRICT out_data, const TensorIdx input_stride,
    const TensorIdx output_stride) const {
  // Get number of elements to be processed in on row
  constexpr auto WIDTH = KernelTrans<FloatType, TYPE>::KN_WIDTH;

  for (TensorUInt ncont_idx = 0; ncont_idx < WIDTH; ++ncont_idx) {
    for (TensorUInt cont_idx = 0; cont_idx < WIDTH; ++cont_idx) {
      const auto input_idx = cont_idx + ncont_idx * input_stride,
          output_idx = ncont_idx + cont_idx * output_stride;
      out_data[output_idx] = this->alpha_ * in_data[input_idx]
          + this->beta_ * out_data[output_idx];
    }
  }
}


template <typename FloatType>
KernelTrans<FloatType, KernelTypeTrans::KERNEL_LINE>::KernelTrans()
    : KernelTransData<FloatType, KernelTypeTrans::KERNEL_LINE>(),
      stride_in_in_(1), stride_in_out_(1), stride_out_in_(1),
      stride_out_out_(1), ld_in_size_(1), ld_out_size_(1) {
}


template <typename FloatType>
void KernelTrans<FloatType, KernelTypeTrans::KERNEL_LINE>::set_wrapper_loop(
    const TensorIdx stride_in_in, const TensorIdx stride_in_out,
    const TensorIdx stride_out_in, const TensorIdx stride_out_out,
    const TensorUInt ld_in_size, const TensorUInt ld_out_size) {
  this->stride_in_in_ = stride_in_in, this->stride_in_out_ = stride_in_out;
  this->stride_out_in_ = stride_out_in, this->stride_out_out_ = stride_out_out;
  this->ld_in_size_ = ld_in_size > 0 ? ld_in_size : 1;
  this->ld_out_size_ = ld_out_size > 0 ? ld_out_size : 1;
}


template <typename FloatType>
void KernelTrans<FloatType, KernelTypeTrans::KERNEL_LINE>::exec(
    const FloatType * RESTRICT in_data, FloatType * RESTRICT out_data,
    const TensorIdx in_size, const TensorIdx out_size) const {
  for (TensorUInt out_idx = 0; out_idx < this->ld_out_size_; ++out_idx) {
    for (TensorUInt in_idx = 0; in_idx < this->ld_in_size_; ++in_idx) {
      auto in_ptr = in_data + this->stride_in_in_ * in_idx
          + this->stride_in_out_ * out_idx;
      auto out_ptr = out_data + this->stride_out_in_ * in_idx
          + this->stride_out_out_ * out_idx;

      for (TensorIdx idx = 0; idx < in_size; ++idx)
        out_ptr[idx] = this->alpha_ * in_ptr[idx] + this->beta_ * out_ptr[idx];
    }
  }
}


/*
 * Explicit template instantiation definition for class KernelTransData
 */
template class KernelTransData<float, KernelTypeTrans::KERNEL_FULL>;
template class KernelTransData<double, KernelTypeTrans::KERNEL_FULL>;
template class KernelTransData<FloatComplex, KernelTypeTrans::KERNEL_FULL>;
template class KernelTransData<DoubleComplex, KernelTypeTrans::KERNEL_FULL>;

template class KernelTransData<float, KernelTypeTrans::KERNEL_HALF>;
template class KernelTransData<double, KernelTypeTrans::KERNEL_HALF>;
template class KernelTransData<FloatComplex, KernelTypeTrans::KERNEL_HALF>;
template class KernelTransData<DoubleComplex, KernelTypeTrans::KERNEL_HALF>;

template class KernelTransData<float, KernelTypeTrans::KERNEL_LINE>;
template class KernelTransData<double, KernelTypeTrans::KERNEL_LINE>;
template class KernelTransData<FloatComplex, KernelTypeTrans::KERNEL_LINE>;
template class KernelTransData<DoubleComplex, KernelTypeTrans::KERNEL_LINE>;


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
