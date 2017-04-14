#include <hptt/kernels/kernel_trans.h>

#include <hptt/types.h>
#include <hptt/util/util_trans.h>
#include <hptt/kernels/macro_kernel_trans.h>


namespace hptt {

/*
 * Implementation for struct KernelPackTrans
 */
template <typename FloatType,
          bool UPDATE_OUT>
void KernelPackTrans<FloatType, UPDATE_OUT>::set_coef(
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
  this->kn_sca_right.set_coef(alpha, beta);
  this->kn_sca_bottom.set_coef(alpha, beta);
  this->kn_sca_scalar.set_coef(alpha, beta);
}


template <typename FloatType,
          bool UPDATE_OUT>
TensorUInt KernelPackTrans<FloatType, UPDATE_OUT>::kernel_offset(
    const KernelTypeTrans kn_type, const TensorUInt cont_size,
    const TensorUInt ncont_size) const {
  if (KernelTypeTrans::KERNEL_FULL == kn_type)
    return 4 * (cont_size - 1) + ncont_size - 1;
  else if (KernelTypeTrans::KERNEL_HALF == kn_type)
    return 16 + (1 == cont_size ? ncont_size - 1 : cont_size + 2);
  else
    return KERNEL_NUM - 3;
}


template <typename FloatType,
          bool UPDATE_OUT>
TensorUInt KernelPackTrans<FloatType, UPDATE_OUT>::kn_cont_len(
    const KernelTypeTrans kn_type) const {
  if (KernelTypeTrans::KERNEL_FULL == kn_type)
    return this->knf_basic.get_cont_len();
  else if (KernelTypeTrans::KERNEL_HALF == kn_type)
    return this->knh_basic.get_cont_len();
  else
    return 1;
}


template <typename FloatType,
          bool UPDATE_OUT>
TensorUInt KernelPackTrans<FloatType, UPDATE_OUT>::kn_ncont_len(
    const KernelTypeTrans kn_type) const {
  if (KernelTypeTrans::KERNEL_FULL == kn_type)
    return this->knf_basic.get_ncont_len();
  else if (KernelTypeTrans::KERNEL_HALF == kn_type)
    return this->knh_basic.get_ncont_len();
  else
    return 1;
}


template <typename FloatType,
          bool UPDATE_OUT>
KernelPackTrans<FloatType, UPDATE_OUT>::KernelPackTrans()
    : linear_loop_max(MacroTransLinear<FloatType, UPDATE_OUT>::LOOP_MAX),
      knf_giant(this->knf_4x4), knf_basic(this->knf_1x1),
      knh_giant(this->knh_1x4), knh_basic(this->knh_1x1) {
}


/*
 * Explicit template instantiation definition for struct KernelPackTrans
 */
template struct KernelPackTrans<float, true>;
template struct KernelPackTrans<double, true>;
template struct KernelPackTrans<FloatComplex, true>;
template struct KernelPackTrans<DoubleComplex, true>;

template struct KernelPackTrans<float, false>;
template struct KernelPackTrans<double, false>;
template struct KernelPackTrans<FloatComplex, false>;
template struct KernelPackTrans<DoubleComplex, false>;

}
