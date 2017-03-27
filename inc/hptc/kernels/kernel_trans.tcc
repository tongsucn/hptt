#pragma once
#ifndef HPTC_KERNELS_KERNEL_TRANS_TCC_
#define HPTC_KERNELS_KERNEL_TRANS_TCC_

/*
 * Implementation for struct KernelPackTrans
 */
template <typename FloatType,
          CoefUsageTrans USAGE>
void KernelPackTrans<FloatType, USAGE>::set_coef(
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
  this->knh_4x1.set_coef(alpha, beta), this->kn_lin.set_coef(alpha, beta);
}


template <typename FloatType,
          CoefUsageTrans USAGE>
TensorUInt KernelPackTrans<FloatType, USAGE>::kernel_offset(
    const KernelTypeTrans kn_type, const TensorUInt cont_size,
    const TensorUInt ncont_size, const bool is_tail) const {
  if (KernelTypeTrans::KERNEL_FULL == kn_type)
    return 4 * (cont_size - 1) + ncont_size - 1;
  else if (KernelTypeTrans::KERNEL_HALF == kn_type)
    return 16 + (1 == cont_size ? ncont_size - 1 : cont_size + 2);
  else
    return KERNEL_NUM - 2 + (is_tail ? 1 : 0);
}


template <typename FloatType,
          CoefUsageTrans USAGE>
TensorUInt KernelPackTrans<FloatType, USAGE>::kn_cont_len(
    const KernelTypeTrans kn_type, const TensorUInt cont_size) const {
  if (KernelTypeTrans::KERNEL_FULL == kn_type)
    return cont_size * this->knf_basic.get_cont_len();
  else if (KernelTypeTrans::KERNEL_HALF == kn_type)
    return cont_size * this->knh_basic.get_cont_len();
  else
    return cont_size;
}


template <typename FloatType,
          CoefUsageTrans USAGE>
TensorUInt KernelPackTrans<FloatType, USAGE>::kn_ncont_len(
    const KernelTypeTrans kn_type, const TensorUInt ncont_size) const {
  if (KernelTypeTrans::KERNEL_FULL == kn_type)
    return ncont_size * this->knf_basic.get_ncont_len();
  else if (KernelTypeTrans::KERNEL_HALF == kn_type)
    return ncont_size * this->knh_basic.get_ncont_len();
  else
    return ncont_size;
}


template <typename FloatType,
          CoefUsageTrans USAGE>
KernelPackTrans<FloatType, USAGE>::KernelPackTrans()
    : knf_giant(this->knf_4x4), knf_basic(this->knf_1x1),
    knh_giant(this->knh_1x4), knh_basic(this->knh_1x1) {
}


/*
 * Explicit instantiation declaration for struct KernelPackTrans
 */
extern template struct KernelPackTrans<float, CoefUsageTrans::USE_NONE>;
extern template struct KernelPackTrans<float, CoefUsageTrans::USE_ALPHA>;
extern template struct KernelPackTrans<float, CoefUsageTrans::USE_BETA>;
extern template struct KernelPackTrans<float, CoefUsageTrans::USE_BOTH>;

extern template struct KernelPackTrans<double, CoefUsageTrans::USE_NONE>;
extern template struct KernelPackTrans<double, CoefUsageTrans::USE_ALPHA>;
extern template struct KernelPackTrans<double, CoefUsageTrans::USE_BETA>;
extern template struct KernelPackTrans<double, CoefUsageTrans::USE_BOTH>;

extern template struct KernelPackTrans<FloatComplex, CoefUsageTrans::USE_NONE>;
extern template struct KernelPackTrans<FloatComplex, CoefUsageTrans::USE_ALPHA>;
extern template struct KernelPackTrans<FloatComplex, CoefUsageTrans::USE_BETA>;
extern template struct KernelPackTrans<FloatComplex, CoefUsageTrans::USE_BOTH>;

extern template struct KernelPackTrans<DoubleComplex, CoefUsageTrans::USE_NONE>;
extern template struct KernelPackTrans<DoubleComplex,
    CoefUsageTrans::USE_ALPHA>;
extern template struct KernelPackTrans<DoubleComplex, CoefUsageTrans::USE_BETA>;
extern template struct KernelPackTrans<DoubleComplex, CoefUsageTrans::USE_BOTH>;

#endif // HPTC_KERNELS_KERNEL_TRANS_TCC_
