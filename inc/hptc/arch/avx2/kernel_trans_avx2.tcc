#pragma once
#ifndef HPTC_ARCH_AVX2_KERNEL_TRANS_AVX2_TCC_
#define HPTC_ARCH_AVX2_KERNEL_TRANS_AVX2_TCC_

/*
 * Implementation of class KernelTransData
 */
template <typename FloatType,
          KernelTypeTrans TYPE>
void KernelTransData<FloatType, TYPE>::set_coef(
    const DeducedFloatType<FloatType> alpha,
    const DeducedFloatType<FloatType> beta) {
  this->alpha_ = alpha, this->beta_ = beta;
  this->reg_alpha_ = DeducedRegType<FloatType, TYPE>::set_reg(this->alpha_);
  this->reg_beta_ = DeducedRegType<FloatType, TYPE>::set_reg(this->beta_);
}


/*
 * Specializations of class KernelTrans
 */
template <>
void KernelTrans<float, KernelTypeTrans::KERNEL_FULL>::exec(
    const float * RESTRICT in_data, float * RESTRICT out_data,
    const TensorIdx input_stride, const TensorIdx output_stride) const;
template <>
void KernelTrans<double, KernelTypeTrans::KERNEL_FULL>::exec(
    const double * RESTRICT in_data, double * RESTRICT out_data,
    const TensorIdx input_stride, const TensorIdx output_stride) const;
template <>
void KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_FULL>::exec(
    const FloatComplex * RESTRICT in_data, FloatComplex * RESTRICT out_data,
    const TensorIdx input_stride, const TensorIdx output_stride) const;
template <>
void KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_FULL>::exec(
    const DoubleComplex * RESTRICT in_data, DoubleComplex * RESTRICT out_data,
    const TensorIdx input_stride, const TensorIdx output_stride) const;


template <>
void KernelTrans<float, KernelTypeTrans::KERNEL_HALF>::exec(
    const float * RESTRICT in_data, float * RESTRICT out_data,
    const TensorIdx input_stride, const TensorIdx output_stride) const;
template <>
void KernelTrans<double, KernelTypeTrans::KERNEL_HALF>::exec(
    const double * RESTRICT in_data, double * RESTRICT out_data,
    const TensorIdx input_stride, const TensorIdx output_stride) const;
template <>
void KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_HALF>::exec(
    const FloatComplex * RESTRICT in_data, FloatComplex * RESTRICT out_data,
    const TensorIdx input_stride, const TensorIdx output_stride) const;
template <>
void KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_HALF>::exec(
    const DoubleComplex * RESTRICT in_data, DoubleComplex * RESTRICT out_data,
    const TensorIdx input_stride, const TensorIdx output_stride) const;


template <typename FloatType>
class KernelTrans<FloatType, KernelTypeTrans::KERNEL_LINE>
    : public KernelTransData<FloatType, KernelTypeTrans::KERNEL_LINE> {
public:
  KernelTrans();

  void set_wrapper_loop(const TensorIdx stride_in_in,
      const TensorIdx stride_in_out, const TensorIdx stride_out_in,
      const TensorIdx stride_out_out, const TensorUInt cont_len,
      const TensorUInt ncont_len);

  void exec(const FloatType * RESTRICT in_data, FloatType * RESTRICT out_data,
      const TensorIdx in_size, const TensorIdx out_size) const;

private:
  TensorIdx stride_in_in_, stride_in_out_, stride_out_in_, stride_out_out_;
  TensorUInt ld_in_, ld_out_;
};


/*
 * Explicit template instantiation declaration for class KernelTrans
 */
extern template class KernelTrans<float, KernelTypeTrans::KERNEL_FULL>;
extern template class KernelTrans<double, KernelTypeTrans::KERNEL_FULL>;
extern template class KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_FULL>;
extern template class KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_FULL>;

extern template class KernelTrans<float, KernelTypeTrans::KERNEL_HALF>;
extern template class KernelTrans<double, KernelTypeTrans::KERNEL_HALF>;
extern template class KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_HALF>;
extern template class KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_HALF>;

extern template class KernelTrans<float, KernelTypeTrans::KERNEL_LINE>;
extern template class KernelTrans<double, KernelTypeTrans::KERNEL_LINE>;
extern template class KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_LINE>;
extern template class KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_LINE>;

#endif // HPTC_ARCH_AVX2_KERNEL_TRANS_AVX2_TCC_
