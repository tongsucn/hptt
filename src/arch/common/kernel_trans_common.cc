#include <hptt/arch/common/kernel_trans_common.h>

#include <hptt/types.h>
#include <hptt/arch/compat.h>
#include <hptt/util/util_trans.h>


namespace hptt {

/*
 * Implementation of class KernelTrans
 */
template <typename FloatType,
          KernelTypeTrans TYPE,
          bool UPDATE_OUT>
KernelTrans<FloatType, TYPE, UPDATE_OUT>::KernelTrans()
    : KernelTransData<FloatType, TYPE>() {
}


template <typename FloatType,
          KernelTypeTrans TYPE,
          bool UPDATE_OUT>
void KernelTrans<FloatType, TYPE, UPDATE_OUT>::exec(
    const FloatType * RESTRICT data_in, FloatType * RESTRICT data_out,
    const TensorIdx stride_in_outld, const TensorIdx stride_out_inld) const {
  // Get number of elements to be processed in on row
  constexpr auto WIDTH = KernelTrans<FloatType, TYPE>::KN_WIDTH;

  if (UPDATE_OUT) {
    for (TensorUInt idx_in_outld = 0; idx_in_outld < WIDTH; ++idx_in_outld) {
      for (TensorUInt idx_in_inld = 0; idx_in_inld < WIDTH; ++idx_in_inld) {
        const auto idx_in = idx_in_inld + idx_in_outld * stride_in_outld,
            idx_out = idx_in_outld + idx_in_inld * stride_out_inld;
        data_out[idx_out] = this->alpha_ * data_in[idx_in]
            + this->beta_ * data_out[idx_out];
      }
    }
  }
  else {
    for (TensorUInt idx_in_outld = 0; idx_in_outld < WIDTH; ++idx_in_outld) {
      for (TensorUInt idx_in_inld = 0; idx_in_inld < WIDTH; ++idx_in_inld) {
        const auto idx_in = idx_in_inld + idx_in_outld * stride_in_outld,
            idx_out = idx_in_outld + idx_in_inld * stride_out_inld;
        data_out[idx_out] = this->alpha_ * data_in[idx_in];
      }
    }
  }
}


template <typename FloatType,
          bool UPDATE_OUT>
KernelTrans<FloatType, KernelTypeTrans::KERNEL_LINE, UPDATE_OUT>::KernelTrans()
    : KernelTransData<FloatType, KernelTypeTrans::KERNEL_LINE>(),
      stride_in_inld_(1), stride_in_outld_(1), stride_out_inld_(1),
      stride_out_outld_(1), size_kn_inld_(1), size_kn_outld_(1) {
}


template <typename FloatType,
          bool UPDATE_OUT>
void KernelTrans<FloatType, KernelTypeTrans::KERNEL_LINE, UPDATE_OUT>::
set_wrapper_loop(const TensorIdx stride_in_inld,
    const TensorIdx stride_in_outld, const TensorIdx stride_out_inld,
    const TensorIdx stride_out_outld, const TensorUInt size_kn_inld,
    const TensorUInt size_kn_outld) {
  this->stride_in_inld_ = stride_in_inld;
  this->stride_in_outld_ = stride_in_outld;
  this->stride_out_inld_ = stride_out_inld;
  this->stride_out_outld_ = stride_out_outld;
  this->size_kn_inld_ = size_kn_inld > 0 ? size_kn_inld : 1;
  this->size_kn_outld_ = size_kn_outld > 0 ? size_kn_outld : 1;
}


template <typename FloatType,
          bool UPDATE_OUT>
void KernelTrans<FloatType, KernelTypeTrans::KERNEL_LINE, UPDATE_OUT>::exec(
    const FloatType * RESTRICT data_in, FloatType * RESTRICT data_out,
    const TensorIdx size_trans, const TensorIdx size_pad) const {
  if (UPDATE_OUT) {
    for (TensorUInt idx_out = 0; idx_out < this->size_kn_outld_; ++idx_out) {
      for (TensorUInt idx_in = 0; idx_in < this->size_kn_inld_; ++idx_in) {
        auto in_ptr = data_in + this->stride_in_inld_ * idx_in
            + this->stride_in_outld_ * idx_out;
        auto out_ptr = data_out + this->stride_out_inld_ * idx_in
            + this->stride_out_outld_ * idx_out;

        for (TensorIdx idx = 0; idx < size_trans; ++idx)
          out_ptr[idx] = this->alpha_ * in_ptr[idx]
              + this->beta_ * out_ptr[idx];
      }
    }
  }
  else {
    for (TensorUInt idx_out = 0; idx_out < this->size_kn_outld_; ++idx_out) {
      for (TensorUInt idx_in = 0; idx_in < this->size_kn_inld_; ++idx_in) {
        const FloatType * RESTRICT in_ptr
            = data_in + this->stride_in_inld_ * idx_in
            + this->stride_in_outld_ * idx_out;
        FloatType * RESTRICT out_ptr
            = data_out + this->stride_out_inld_ * idx_in
            + this->stride_out_outld_ * idx_out;

        for (TensorIdx idx = 0; idx < size_trans; ++idx)
          out_ptr[idx] = this->alpha_ * in_ptr[idx];
      }
    }
  }
}


/*
 * Explicit template instantiation definition for class KernelTrans
 */
template class KernelTrans<float, KernelTypeTrans::KERNEL_FULL, true>;
template class KernelTrans<double, KernelTypeTrans::KERNEL_FULL, true>;
template class KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_FULL, true>;
template class KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_FULL, true>;

template class KernelTrans<float, KernelTypeTrans::KERNEL_HALF, true>;
template class KernelTrans<double, KernelTypeTrans::KERNEL_HALF, true>;
template class KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_HALF, true>;
template class KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_HALF, true>;

template class KernelTrans<float, KernelTypeTrans::KERNEL_LINE, true>;
template class KernelTrans<double, KernelTypeTrans::KERNEL_LINE, true>;
template class KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_LINE, true>;
template class KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_LINE, true>;

template class KernelTrans<float, KernelTypeTrans::KERNEL_FULL, false>;
template class KernelTrans<double, KernelTypeTrans::KERNEL_FULL, false>;
template class KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_FULL, false>;
template class KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_FULL, false>;

template class KernelTrans<float, KernelTypeTrans::KERNEL_HALF, false>;
template class KernelTrans<double, KernelTypeTrans::KERNEL_HALF, false>;
template class KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_HALF, false>;
template class KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_HALF, false>;

template class KernelTrans<float, KernelTypeTrans::KERNEL_LINE, false>;
template class KernelTrans<double, KernelTypeTrans::KERNEL_LINE, false>;
template class KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_LINE, false>;
template class KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_LINE, false>;

}
