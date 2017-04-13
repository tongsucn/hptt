#include <hptt/arch/arm/kernel_trans_arm.h>

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
  constexpr auto WIDTH = KernelTrans<FloatType, TYPE, UPDATE_OUT>::KN_WIDTH;

  if (UPDATE_OUT) {
#pragma omp simd collapse(2)
    for (TensorUInt idx_row = 0; idx_row < WIDTH; ++idx_row) {
      for (TensorUInt idx_col = 0; idx_col < WIDTH; ++idx_col) {
        const TensorIdx offset_in = idx_col + idx_row * stride_out_inld;
        const TensorIdx offset_out = idx_row + idx_col * stride_out_inld;
        data_out[offset_out] = this->reg_alpha_ * data_in[offset_in]
            + this->reg_beta_ * data_out[offset_out];
      }
    }
  }
  else {
#pragma omp simd collapse(2)
    for (TensorUInt idx_row = 0; idx_row < WIDTH; ++idx_row) {
      for (TensorUInt idx_col = 0; idx_col < WIDTH; ++idx_col) {
        const TensorIdx offset_in = idx_col + idx_row * stride_out_inld;
        const TensorIdx offset_out = idx_row + idx_col * stride_out_inld;
        data_out[offset_out] = this->reg_alpha_ * data_in[offset_in];
      }
    }
  }
}


template <typename FloatType,
          bool UPDATE_OUT>
KernelTrans<FloatType, KernelTypeTrans::KERNEL_LINE, UPDATE_OUT>::KernelTrans()
    : KernelTransData<FloatType, KernelTypeTrans::KERNEL_LINE>() {
}


template <typename FloatType,
          bool UPDATE_OUT>
void KernelTrans<FloatType, KernelTypeTrans::KERNEL_LINE, UPDATE_OUT>::exec(
    const FloatType * RESTRICT data_in, FloatType * RESTRICT data_out,
    const TensorIdx size_trans, const TensorIdx size_pad) const {
  if (UPDATE_OUT)
#pragma omp simd
    for (TensorIdx idx = 0; idx < size_trans; ++idx)
      data_out[idx] = this->alpha_ * data_in[idx] + this->beta_ * data_out[idx];
  else
#pragma omp simd
    for (TensorIdx idx = 0; idx < size_trans; ++idx)
      data_out[idx] = this->alpha_ * data_in[idx];
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
