#include <hptc/kernels/kernel_trans.h>

#include <hptc/types.h>
#include <hptc/util/util_trans.h>
#include <hptc/kernels/macro_kernel_trans.h>


namespace hptc {

/*
 * Implementation for struct KernelPackTrans
 */
template <typename FloatType>
void KernelPackTrans<FloatType>::set_coef(
    const DeducedFloatType<FloatType> alpha,
    const DeducedFloatType<FloatType> beta) {
  this->knf_1x1.set_coef(alpha, beta), this->knf_1x2.set_coef(alpha, beta);
  this->knf_1x3.set_coef(alpha, beta), this->knf_1x4.set_coef(alpha, beta);
  this->knf_2x1.set_coef(alpha, beta), this->knf_2x2.set_coef(alpha, beta);
  this->knf_2x3.set_coef(alpha, beta), this->knf_2x4.set_coef(alpha, beta);
  this->knf_3x1.set_coef(alpha, beta), this->knf_3x2.set_coef(alpha, beta);
  this->knf_3x3.set_coef(alpha, beta), this->knf_3x4.set_coef(alpha, beta);
  this->knf_4x1.set_coef(alpha, beta), this->knf_4x2.set_coef(alpha, beta);
  this->knf_4x3.set_coef(alpha, beta), this->knf_4x4.set_coef(alpha, beta);

  this->knh_1x1.set_coef(alpha, beta), this->knh_1x2.set_coef(alpha, beta);
  this->knh_1x3.set_coef(alpha, beta), this->knh_1x4.set_coef(alpha, beta);
  this->knh_2x1.set_coef(alpha, beta), this->knh_3x1.set_coef(alpha, beta);
  this->knh_4x1.set_coef(alpha, beta), this->kn_lin_core.set_coef(alpha, beta);
  this->kn_lin_right.set_coef(alpha, beta);
  this->kn_lin_bottom.set_coef(alpha, beta);
  this->kn_lin_scalar.set_coef(alpha, beta);
  this->kn_scl.set_coef(alpha, beta);
}


template <typename FloatType>
TensorUInt KernelPackTrans<FloatType>::kernel_offset(
    const KernelTypeTrans kn_type, const TensorUInt cont_size,
    const TensorUInt ncont_size, const bool is_tail) const {
  if (KernelTypeTrans::KERNEL_FULL == kn_type)
    return 4 * (cont_size - 1) + ncont_size - 1;
  else if (KernelTypeTrans::KERNEL_HALF == kn_type)
    return 16 + (1 == cont_size ? ncont_size - 1 : cont_size + 2);
  else
    return KERNEL_NUM - 2 + (is_tail ? 1 : 0);
}


template <typename FloatType>
TensorUInt KernelPackTrans<FloatType>::kn_cont_len(
    const KernelTypeTrans kn_type) const {
  if (KernelTypeTrans::KERNEL_FULL == kn_type)
    return this->knf_basic.get_cont_len();
  else if (KernelTypeTrans::KERNEL_HALF == kn_type)
    return this->knh_basic.get_cont_len();
  else
    return 1;
}


template <typename FloatType>
TensorUInt KernelPackTrans<FloatType>::kn_ncont_len(
    const KernelTypeTrans kn_type) const {
  if (KernelTypeTrans::KERNEL_FULL == kn_type)
    return this->knf_basic.get_ncont_len();
  else if (KernelTypeTrans::KERNEL_HALF == kn_type)
    return this->knh_basic.get_ncont_len();
  else
    return 1;
}


template <typename FloatType>
KernelPackTrans<FloatType>::KernelPackTrans()
    : linear_loop_max(MacroTransLinear<FloatType>::LOOP_MAX),
      knf_giant(this->knf_4x4), knf_basic(this->knf_1x1),
      knh_giant(this->knh_1x4), knh_basic(this->knh_1x1) {
}


/*
 * Explicit template instantiation definition for struct KernelPackTrans
 */
template struct KernelPackTrans<float>;
template struct KernelPackTrans<double>;
template struct KernelPackTrans<FloatComplex>;
template struct KernelPackTrans<DoubleComplex>;

}
