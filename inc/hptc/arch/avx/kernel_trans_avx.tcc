#pragma once
#ifndef HPTC_ARCH_AVX_KERNEL_TRANS_AVX_TCC_
#define HPTC_ARCH_AVX_KERNEL_TRANS_AVX_TCC_

/*
 * Specializations of class KernelTrans
 */
template <>
void KernelTrans<float, KernelTypeTrans::KERNEL_FULL>::set_coef(
    const DeducedFloatType<float> alpha, const DeducedFloatType<float> beta);
template <>
void KernelTrans<float, KernelTypeTrans::KERNEL_FULL>::exec(
    const float * RESTRICT in_data, float * RESTRICT out_data,
    const TensorIdx input_stride, const TensorIdx output_stride) const;

template <>
void KernelTrans<double, KernelTypeTrans::KERNEL_FULL>::set_coef(
    const DeducedFloatType<double> alpha, const DeducedFloatType<double> beta);
template <>
void KernelTrans<double, KernelTypeTrans::KERNEL_FULL>::exec(
    const double * RESTRICT in_data, double * RESTRICT out_data,
    const TensorIdx input_stride, const TensorIdx output_stride) const;

template <>
void KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_FULL>::set_coef(
    const DeducedFloatType<FloatComplex> alpha,
    const DeducedFloatType<FloatComplex> beta);
template <>
void KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_FULL>::exec(
    const FloatComplex * RESTRICT in_data, FloatComplex * RESTRICT out_data,
    const TensorIdx input_stride, const TensorIdx output_stride) const;

template <>
void KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_FULL>::set_coef(
    const DeducedFloatType<DoubleComplex> alpha,
    const DeducedFloatType<DoubleComplex> beta);
template <>
void KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_FULL>::exec(
    const DoubleComplex * RESTRICT in_data, DoubleComplex * RESTRICT out_data,
    const TensorIdx input_stride, const TensorIdx output_stride) const;


template <>
void KernelTrans<float, KernelTypeTrans::KERNEL_HALF>::set_coef(
    const DeducedFloatType<float> alpha, const DeducedFloatType<float> beta);
template <>
void KernelTrans<float, KernelTypeTrans::KERNEL_HALF>::exec(
    const float * RESTRICT in_data, float * RESTRICT out_data,
    const TensorIdx input_stride, const TensorIdx output_stride) const;

template <>
void KernelTrans<double, KernelTypeTrans::KERNEL_HALF>::set_coef(
    const DeducedFloatType<double> alpha, const DeducedFloatType<double> beta);
template <>
void KernelTrans<double, KernelTypeTrans::KERNEL_HALF>::exec(
    const double * RESTRICT in_data, double * RESTRICT out_data,
    const TensorIdx input_stride, const TensorIdx output_stride) const;

template <>
void KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_HALF>::set_coef(
    const DeducedFloatType<FloatComplex> alpha,
    const DeducedFloatType<FloatComplex> beta);
template <>
void KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_HALF>::exec(
    const FloatComplex * RESTRICT in_data, FloatComplex * RESTRICT out_data,
    const TensorIdx input_stride, const TensorIdx output_stride) const;

template <>
void KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_HALF>::set_coef(
    const DeducedFloatType<DoubleComplex> alpha,
    const DeducedFloatType<DoubleComplex> beta);
template <>
void KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_HALF>::exec(
    const DoubleComplex * RESTRICT in_data, DoubleComplex * RESTRICT out_data,
    const TensorIdx input_stride, const TensorIdx output_stride) const;


template <typename FloatType>
class KernelTrans<FloatType, KernelTypeTrans::KERNEL_LINE>
    : public KernelTransData<FloatType, KernelTypeTrans::KERNEL_LINE> {
public:
  void set_coef(const DeducedFloatType<FloatType> alpha,
      const DeducedFloatType<FloatType> beta);

  void exec(const FloatType * RESTRICT in_data, FloatType * RESTRICT out_data,
      const TensorIdx input_stride, const TensorIdx output_stride) const;
};


/*
 * Explicit template instantiation declaration for class KernelTransData and
 * KernelTrans
 */
extern template class KernelTransData<float, KernelTypeTrans::KERNEL_FULL>;
extern template class KernelTransData<double, KernelTypeTrans::KERNEL_FULL>;
extern template class KernelTransData<FloatComplex,
    KernelTypeTrans::KERNEL_FULL>;
extern template class KernelTransData<DoubleComplex,
    KernelTypeTrans::KERNEL_FULL>;

extern template class KernelTransData<float, KernelTypeTrans::KERNEL_HALF>;
extern template class KernelTransData<double, KernelTypeTrans::KERNEL_HALF>;
extern template class KernelTransData<FloatComplex,
    KernelTypeTrans::KERNEL_HALF>;
extern template class KernelTransData<DoubleComplex,
    KernelTypeTrans::KERNEL_HALF>;

extern template class KernelTransData<float, KernelTypeTrans::KERNEL_LINE>;
extern template class KernelTransData<double, KernelTypeTrans::KERNEL_LINE>;
extern template class KernelTransData<FloatComplex,
    KernelTypeTrans::KERNEL_LINE>;
extern template class KernelTransData<DoubleComplex,
    KernelTypeTrans::KERNEL_LINE>;


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

#endif // HPTC_ARCH_AVX_KERNEL_TRANS_AVX_TCC_
