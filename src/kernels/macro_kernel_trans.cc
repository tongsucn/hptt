#include <hptc/kernels/macro_kernel_trans.h>


namespace hptc {

/*
 * Implementation for class MacroTransVec
 */
template <typename KernelFunc>
class MacroTransVec<KernelFunc, 0, 0> {
};


template <typename KernelFunc,
          GenNumType CONT_LEN,
          GenNumType NCONT_LEN>
MacroTransVec<KernelFunc, CONT_LEN, NCONT_LEN>::MacroTransVec(
    DeducedFloatType<typename KernelFunc::FLOAT> alpha,
    DeducedFloatType<typename KernelFunc::FLOAT> beta)
    : kernel_(alpha, beta),
      kn_wd_(this->kernel_.get_kernel_width()) {
}


template <typename KernelFunc,
          GenNumType CONT_LEN,
          GenNumType NCONT_LEN>
GenNumType MacroTransVec<KernelFunc, CONT_LEN, NCONT_LEN>::get_cont_len() {
  return CONT_LEN * this->kn_wd_;
}


template <typename KernelFunc,
          GenNumType CONT_LEN,
          GenNumType NCONT_LEN>
GenNumType MacroTransVec<KernelFunc, CONT_LEN, NCONT_LEN>::get_ncont_len() {
  return NCONT_LEN * this->kn_wd_;
}


template <typename KernelFunc,
          GenNumType CONT_LEN,
          GenNumType NCONT_LEN>
void MacroTransVec<KernelFunc, CONT_LEN, NCONT_LEN>::operator()(
    const typename KernelFunc::FLOAT * RESTRICT input_data,
    typename KernelFunc::FLOAT * RESTRICT output_data,
    const TensorIdx input_stride, const TensorIdx output_stride) {
  this->ncont_tiler_(DualCounter<CONT_LEN, NCONT_LEN>(),
      input_data, output_data, input_stride, output_stride);
}


template <typename KernelFunc,
          GenNumType CONT_LEN,
          GenNumType NCONT_LEN>
template <GenNumType CONT,
         GenNumType NCONT>
void MacroTransVec<KernelFunc, CONT_LEN, NCONT_LEN>::ncont_tiler_(
    DualCounter<CONT, NCONT>,
    const typename KernelFunc::FLOAT * RESTRICT input_data,
    typename KernelFunc::FLOAT * RESTRICT output_data,
    const TensorIdx input_stride, const TensorIdx output_stride) {
  this->ncont_tiler_(DualCounter<CONT, NCONT - 1>(), input_data, output_data,
      input_stride, output_stride);
  this->cont_tiler_(DualCounter<CONT - 1, NCONT - 1>(), input_data, output_data,
      input_stride, output_stride);
}


template <typename KernelFunc,
          GenNumType CONT_LEN,
          GenNumType NCONT_LEN>
template <GenNumType CONT>
void MacroTransVec<KernelFunc, CONT_LEN, NCONT_LEN>::ncont_tiler_(
    DualCounter<CONT, 0>,
    const typename KernelFunc::FLOAT * RESTRICT input_data,
    typename KernelFunc::FLOAT * RESTRICT output_data,
    const TensorIdx input_stride, const TensorIdx output_stride) {
}


template <typename KernelFunc,
          GenNumType CONT_LEN,
          GenNumType NCONT_LEN>
template <GenNumType CONT,
         GenNumType NCONT>
void MacroTransVec<KernelFunc, CONT_LEN, NCONT_LEN>::cont_tiler_(
    DualCounter<CONT, NCONT>,
    const typename KernelFunc::FLOAT * RESTRICT input_data,
    typename KernelFunc::FLOAT * RESTRICT output_data,
    const TensorIdx input_stride, const TensorIdx output_stride) {
  this->cont_tiler_(DualCounter<CONT - 1, NCONT>(), input_data, output_data,
      input_stride, output_stride);
  this->kernel_(
      input_data + CONT * this->kn_wd_ + NCONT * this->kn_wd_ * input_stride,
      output_data + NCONT * this->kn_wd_ + CONT * this->kn_wd_ * output_stride,
      input_stride, output_stride);
}


template <typename KernelFunc,
          GenNumType CONT_LEN,
          GenNumType NCONT_LEN>
template <GenNumType NCONT>
void MacroTransVec<KernelFunc, CONT_LEN, NCONT_LEN>::cont_tiler_(
    DualCounter<0, NCONT>,
    const typename KernelFunc::FLOAT * RESTRICT input_data,
    typename KernelFunc::FLOAT * RESTRICT output_data,
    const TensorIdx input_stride, const TensorIdx output_stride) {
  this->kernel_(input_data + NCONT * this->kn_wd_ * input_stride,
      output_data + NCONT * this->kn_wd_, input_stride, output_stride);
}


/*
 * Implementation for class MacroTransMemcpy
 */
template <typename FloatType>
MacroTransMemcpy<FloatType>::MacroTransMemcpy(DeducedFloatType<FloatType> alpha,
    DeducedFloatType<FloatType> beta)
    : alpha(alpha), beta(beta) {
}


template <typename FloatType>
void MacroTransMemcpy<FloatType>::operator()(
    const FloatType * RESTRICT input_data, FloatType * RESTRICT output_data,
    const TensorIdx input_stride, const TensorIdx output_stride) {
  std::copy(input_data, input_data + input_stride, output_data);
}


/*
 * Implementation for class MacroTransScalar
 */
template <typename FloatType,
          CoefUsageTrans USAGE>
MacroTransScalar<FloatType, USAGE>::MacroTransScalar(
    DeducedFloatType<FloatType> alpha, DeducedFloatType<FloatType> beta)
    : alpha(alpha), beta(beta) {
}


template <typename FloatType,
          CoefUsageTrans USAGE>
void MacroTransScalar<FloatType, USAGE>::operator()(
    const FloatType * RESTRICT input_data, FloatType * RESTRICT output_data,
    const TensorIdx input_stride, const TensorIdx output_stride) {
  if (USAGE == CoefUsageTrans::USE_NONE)
    *output_data = *input_data;
  else if (USAGE == CoefUsageTrans::USE_ALPHA)
    *output_data = this->alpha * (*input_data);
  else if (USAGE == CoefUsageTrans::USE_BETA)
    *output_data = *input_data + this->beta * (*output_data);
  else
    *output_data = this->alpha * (*input_data) + this->beta * (*output_data);
}


/*
 * Explicit template instantiation for class MacroTransVec
 */
template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_NONE>, 4, 4>;
template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_ALPHA>, 4, 4>;
template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_BETA>, 4, 4>;
template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_BOTH>, 4, 4>;
template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_NONE>, 4, 4>;
template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_ALPHA>, 4, 4>;
template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_BETA>, 4, 4>;
template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_BOTH>, 4, 4>;
template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_NONE>, 4, 4>;
template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_ALPHA>, 4, 4>;
template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_BETA>, 4, 4>;
template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_BOTH>, 4, 4>;
template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_NONE>, 4, 4>;
template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_ALPHA>, 4, 4>;
template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_BETA>, 4, 4>;
template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_BOTH>, 4, 4>;
template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_NONE>, 1, 4>;
template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_ALPHA>, 1, 4>;
template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_BETA>, 1, 4>;
template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_BOTH>, 1, 4>;
template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_NONE>, 1, 4>;
template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_ALPHA>, 1, 4>;
template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_BETA>, 1, 4>;
template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_BOTH>, 1, 4>;
template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_NONE>, 1, 4>;
template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_ALPHA>, 1, 4>;
template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_BETA>, 1, 4>;
template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_BOTH>, 1, 4>;
template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_NONE>, 1, 4>;
template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_ALPHA>, 1, 4>;
template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_BETA>, 1, 4>;
template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_BOTH>, 1, 4>;
template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_NONE>, 4, 1>;
template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_ALPHA>, 4, 1>;
template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_BETA>, 4, 1>;
template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_BOTH>, 4, 1>;
template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_NONE>, 4, 1>;
template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_ALPHA>, 4, 1>;
template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_BETA>, 4, 1>;
template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_BOTH>, 4, 1>;
template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_NONE>, 4, 1>;
template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_ALPHA>, 4, 1>;
template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_BETA>, 4, 1>;
template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_BOTH>, 4, 1>;
template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_NONE>, 4, 1>;
template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_ALPHA>, 4, 1>;
template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_BETA>, 4, 1>;
template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_BOTH>, 4, 1>;
template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_NONE>, 1, 1>;
template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_ALPHA>, 1, 1>;
template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_BETA>, 1, 1>;
template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_BOTH>, 1, 1>;
template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_NONE>, 1, 1>;
template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_ALPHA>, 1, 1>;
template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_BETA>, 1, 1>;
template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_BOTH>, 1, 1>;
template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_NONE>, 1, 1>;
template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_ALPHA>, 1, 1>;
template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_BETA>, 1, 1>;
template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_BOTH>, 1, 1>;
template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_NONE>, 1, 1>;
template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_ALPHA>, 1, 1>;
template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_BETA>, 1, 1>;
template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_BOTH>, 1, 1>;

template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_NONE>, 1, 1>;
template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_ALPHA>, 1, 1>;
template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_BETA>, 1, 1>;
template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_BOTH>, 1, 1>;
template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_NONE>, 1, 1>;
template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_ALPHA>, 1, 1>;
template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_BETA>, 1, 1>;
template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_BOTH>, 1, 1>;
template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_NONE>, 1, 1>;
template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_ALPHA>, 1, 1>;
template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_BETA>, 1, 1>;
template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_BOTH>, 1, 1>;
template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_NONE>, 1, 1>;
template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_ALPHA>, 1, 1>;
template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_BETA>, 1, 1>;
template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_BOTH>, 1, 1>;
template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_NONE>, 1, 2>;
template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_ALPHA>, 1, 2>;
template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_BETA>, 1, 2>;
template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_BOTH>, 1, 2>;
template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_NONE>, 1, 2>;
template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_ALPHA>, 1, 2>;
template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_BETA>, 1, 2>;
template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_BOTH>, 1, 2>;
template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_NONE>, 1, 2>;
template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_ALPHA>, 1, 2>;
template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_BETA>, 1, 2>;
template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_BOTH>, 1, 2>;
template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_NONE>, 1, 2>;
template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_ALPHA>, 1, 2>;
template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_BETA>, 1, 2>;
template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_BOTH>, 1, 2>;
template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_NONE>, 2, 1>;
template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_ALPHA>, 2, 1>;
template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_BETA>, 2, 1>;
template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_BOTH>, 2, 1>;
template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_NONE>, 2, 1>;
template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_ALPHA>, 2, 1>;
template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_BETA>, 2, 1>;
template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_BOTH>, 2, 1>;
template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_NONE>, 2, 1>;
template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_ALPHA>, 2, 1>;
template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_BETA>, 2, 1>;
template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_BOTH>, 2, 1>;
template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_NONE>, 2, 1>;
template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_ALPHA>, 2, 1>;
template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_BETA>, 2, 1>;
template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_BOTH>, 2, 1>;

template class MacroTransVec<
    KernelTransLinear<float, CoefUsageTrans::USE_NONE>, 8, 1>;
template class MacroTransVec<
    KernelTransLinear<float, CoefUsageTrans::USE_ALPHA>, 8, 1>;
template class MacroTransVec<
    KernelTransLinear<float, CoefUsageTrans::USE_BETA>, 8, 1>;
template class MacroTransVec<
    KernelTransLinear<float, CoefUsageTrans::USE_BOTH>, 8, 1>;
template class MacroTransVec<
    KernelTransLinear<double, CoefUsageTrans::USE_NONE>, 8, 1>;
template class MacroTransVec<
    KernelTransLinear<double, CoefUsageTrans::USE_ALPHA>, 8, 1>;
template class MacroTransVec<
    KernelTransLinear<double, CoefUsageTrans::USE_BETA>, 8, 1>;
template class MacroTransVec<
    KernelTransLinear<double, CoefUsageTrans::USE_BOTH>, 8, 1>;
template class MacroTransVec<
    KernelTransLinear<FloatComplex, CoefUsageTrans::USE_NONE>, 8, 1>;
template class MacroTransVec<
    KernelTransLinear<FloatComplex, CoefUsageTrans::USE_ALPHA>, 8, 1>;
template class MacroTransVec<
    KernelTransLinear<FloatComplex, CoefUsageTrans::USE_BETA>, 8, 1>;
template class MacroTransVec<
    KernelTransLinear<FloatComplex, CoefUsageTrans::USE_BOTH>, 8, 1>;
template class MacroTransVec<
    KernelTransLinear<DoubleComplex, CoefUsageTrans::USE_NONE>, 8, 1>;
template class MacroTransVec<
    KernelTransLinear<DoubleComplex, CoefUsageTrans::USE_ALPHA>, 8, 1>;
template class MacroTransVec<
    KernelTransLinear<DoubleComplex, CoefUsageTrans::USE_BETA>, 8, 1>;
template class MacroTransVec<
    KernelTransLinear<DoubleComplex, CoefUsageTrans::USE_BOTH>, 8, 1>;
template class MacroTransVec<
    KernelTransLinear<float, CoefUsageTrans::USE_NONE>, 4, 1>;
template class MacroTransVec<
    KernelTransLinear<float, CoefUsageTrans::USE_ALPHA>, 4, 1>;
template class MacroTransVec<
    KernelTransLinear<float, CoefUsageTrans::USE_BETA>, 4, 1>;
template class MacroTransVec<
    KernelTransLinear<float, CoefUsageTrans::USE_BOTH>, 4, 1>;
template class MacroTransVec<
    KernelTransLinear<double, CoefUsageTrans::USE_NONE>, 4, 1>;
template class MacroTransVec<
    KernelTransLinear<double, CoefUsageTrans::USE_ALPHA>, 4, 1>;
template class MacroTransVec<
    KernelTransLinear<double, CoefUsageTrans::USE_BETA>, 4, 1>;
template class MacroTransVec<
    KernelTransLinear<double, CoefUsageTrans::USE_BOTH>, 4, 1>;
template class MacroTransVec<
    KernelTransLinear<FloatComplex, CoefUsageTrans::USE_NONE>, 4, 1>;
template class MacroTransVec<
    KernelTransLinear<FloatComplex, CoefUsageTrans::USE_ALPHA>, 4, 1>;
template class MacroTransVec<
    KernelTransLinear<FloatComplex, CoefUsageTrans::USE_BETA>, 4, 1>;
template class MacroTransVec<
    KernelTransLinear<FloatComplex, CoefUsageTrans::USE_BOTH>, 4, 1>;
template class MacroTransVec<
    KernelTransLinear<DoubleComplex, CoefUsageTrans::USE_NONE>, 4, 1>;
template class MacroTransVec<
    KernelTransLinear<DoubleComplex, CoefUsageTrans::USE_ALPHA>, 4, 1>;
template class MacroTransVec<
    KernelTransLinear<DoubleComplex, CoefUsageTrans::USE_BETA>, 4, 1>;
template class MacroTransVec<
    KernelTransLinear<DoubleComplex, CoefUsageTrans::USE_BOTH>, 4, 1>;
template class MacroTransVec<
    KernelTransLinear<float, CoefUsageTrans::USE_NONE>, 2, 1>;
template class MacroTransVec<
    KernelTransLinear<float, CoefUsageTrans::USE_ALPHA>, 2, 1>;
template class MacroTransVec<
    KernelTransLinear<float, CoefUsageTrans::USE_BETA>, 2, 1>;
template class MacroTransVec<
    KernelTransLinear<float, CoefUsageTrans::USE_BOTH>, 2, 1>;
template class MacroTransVec<
    KernelTransLinear<double, CoefUsageTrans::USE_NONE>, 2, 1>;
template class MacroTransVec<
    KernelTransLinear<double, CoefUsageTrans::USE_ALPHA>, 2, 1>;
template class MacroTransVec<
    KernelTransLinear<double, CoefUsageTrans::USE_BETA>, 2, 1>;
template class MacroTransVec<
    KernelTransLinear<double, CoefUsageTrans::USE_BOTH>, 2, 1>;
template class MacroTransVec<
    KernelTransLinear<FloatComplex, CoefUsageTrans::USE_NONE>, 2, 1>;
template class MacroTransVec<
    KernelTransLinear<FloatComplex, CoefUsageTrans::USE_ALPHA>, 2, 1>;
template class MacroTransVec<
    KernelTransLinear<FloatComplex, CoefUsageTrans::USE_BETA>, 2, 1>;
template class MacroTransVec<
    KernelTransLinear<FloatComplex, CoefUsageTrans::USE_BOTH>, 2, 1>;
template class MacroTransVec<
    KernelTransLinear<DoubleComplex, CoefUsageTrans::USE_NONE>, 2, 1>;
template class MacroTransVec<
    KernelTransLinear<DoubleComplex, CoefUsageTrans::USE_ALPHA>, 2, 1>;
template class MacroTransVec<
    KernelTransLinear<DoubleComplex, CoefUsageTrans::USE_BETA>, 2, 1>;
template class MacroTransVec<
    KernelTransLinear<DoubleComplex, CoefUsageTrans::USE_BOTH>, 2, 1>;
template class MacroTransVec<
    KernelTransLinear<float, CoefUsageTrans::USE_NONE>, 1, 1>;
template class MacroTransVec<
    KernelTransLinear<float, CoefUsageTrans::USE_ALPHA>, 1, 1>;
template class MacroTransVec<
    KernelTransLinear<float, CoefUsageTrans::USE_BETA>, 1, 1>;
template class MacroTransVec<
    KernelTransLinear<float, CoefUsageTrans::USE_BOTH>, 1, 1>;
template class MacroTransVec<
    KernelTransLinear<double, CoefUsageTrans::USE_NONE>, 1, 1>;
template class MacroTransVec<
    KernelTransLinear<double, CoefUsageTrans::USE_ALPHA>, 1, 1>;
template class MacroTransVec<
    KernelTransLinear<double, CoefUsageTrans::USE_BETA>, 1, 1>;
template class MacroTransVec<
    KernelTransLinear<double, CoefUsageTrans::USE_BOTH>, 1, 1>;
template class MacroTransVec<
    KernelTransLinear<FloatComplex, CoefUsageTrans::USE_NONE>, 1, 1>;
template class MacroTransVec<
    KernelTransLinear<FloatComplex, CoefUsageTrans::USE_ALPHA>, 1, 1>;
template class MacroTransVec<
    KernelTransLinear<FloatComplex, CoefUsageTrans::USE_BETA>, 1, 1>;
template class MacroTransVec<
    KernelTransLinear<FloatComplex, CoefUsageTrans::USE_BOTH>, 1, 1>;
template class MacroTransVec<
    KernelTransLinear<DoubleComplex, CoefUsageTrans::USE_NONE>, 1, 1>;
template class MacroTransVec<
    KernelTransLinear<DoubleComplex, CoefUsageTrans::USE_ALPHA>, 1, 1>;
template class MacroTransVec<
    KernelTransLinear<DoubleComplex, CoefUsageTrans::USE_BETA>, 1, 1>;
template class MacroTransVec<
    KernelTransLinear<DoubleComplex, CoefUsageTrans::USE_BOTH>, 1, 1>;


/*
 * Explicit template instantiation for class MacroTransMemcpy
 */
template class MacroTransMemcpy<float>;
template class MacroTransMemcpy<double>;
template class MacroTransMemcpy<FloatComplex>;
template class MacroTransMemcpy<DoubleComplex>;


/*
 * Explicit template instantiation for class MacroTransScalar
 */
template class MacroTransScalar<float, CoefUsageTrans::USE_NONE>;
template class MacroTransScalar<float, CoefUsageTrans::USE_ALPHA>;
template class MacroTransScalar<float, CoefUsageTrans::USE_BETA>;
template class MacroTransScalar<float, CoefUsageTrans::USE_BOTH>;
template class MacroTransScalar<double, CoefUsageTrans::USE_NONE>;
template class MacroTransScalar<double, CoefUsageTrans::USE_ALPHA>;
template class MacroTransScalar<double, CoefUsageTrans::USE_BETA>;
template class MacroTransScalar<double, CoefUsageTrans::USE_BOTH>;
template class MacroTransScalar<FloatComplex, CoefUsageTrans::USE_NONE>;
template class MacroTransScalar<FloatComplex, CoefUsageTrans::USE_ALPHA>;
template class MacroTransScalar<FloatComplex, CoefUsageTrans::USE_BETA>;
template class MacroTransScalar<FloatComplex, CoefUsageTrans::USE_BOTH>;
template class MacroTransScalar<DoubleComplex, CoefUsageTrans::USE_NONE>;
template class MacroTransScalar<DoubleComplex, CoefUsageTrans::USE_ALPHA>;
template class MacroTransScalar<DoubleComplex, CoefUsageTrans::USE_BETA>;
template class MacroTransScalar<DoubleComplex, CoefUsageTrans::USE_BOTH>;

}
