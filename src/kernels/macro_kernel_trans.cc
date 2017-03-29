#include <hptc/kernels/macro_kernel_trans.h>

#include <algorithm>

#include <hptc/types.h>
#include <hptc/util/util_trans.h>


namespace hptc {

/*
 * Specialization and implementation for class MacroTrans
 */
template <typename MicroKernel>
class MacroTrans<MicroKernel, 0, 0> {
};


template <typename MicroKernel,
          TensorUInt CONT_LEN,
          TensorUInt NCONT_LEN>
MacroTrans<MicroKernel, CONT_LEN, NCONT_LEN>::MacroTrans()
    : kernel_(),
      kn_width_(this->kernel_.kn_width()) {
}


template <typename MicroKernel,
          TensorUInt CONT_LEN,
          TensorUInt NCONT_LEN>
void MacroTrans<MicroKernel, CONT_LEN, NCONT_LEN>::set_coef(
    const DeducedFloatType<typename MicroKernel::Float> alpha,
    const DeducedFloatType<typename MicroKernel::Float> beta) {
  this->kernel_.set_coef(alpha, beta);
}


template <typename MicroKernel,
          TensorUInt CONT_LEN,
          TensorUInt NCONT_LEN>
TensorUInt MacroTrans<MicroKernel, CONT_LEN, NCONT_LEN>::get_cont_len(
    ) const {
  return CONT_LEN * this->kn_width_;
}


template <typename MicroKernel,
          TensorUInt CONT_LEN,
          TensorUInt NCONT_LEN>
TensorUInt MacroTrans<MicroKernel, CONT_LEN, NCONT_LEN>::get_ncont_len(
    ) const {
  return NCONT_LEN * this->kn_width_;
}


template <typename MicroKernel,
          TensorUInt CONT_LEN,
          TensorUInt NCONT_LEN>
void MacroTrans<MicroKernel, CONT_LEN, NCONT_LEN>::exec(
    const typename MicroKernel::Float * RESTRICT input_data,
    typename MicroKernel::Float * RESTRICT output_data,
    const TensorIdx input_stride, const TensorIdx output_stride) const {
  this->ncont_tiler_(DualCounter<CONT_LEN, NCONT_LEN>(), input_data,
      output_data, input_stride, output_stride);
}


template <typename MicroKernel,
          TensorUInt CONT_LEN,
          TensorUInt NCONT_LEN>
template <TensorUInt CONT,
         TensorUInt NCONT>
void MacroTrans<MicroKernel, CONT_LEN, NCONT_LEN>::ncont_tiler_(
    DualCounter<CONT, NCONT>,
    const typename MicroKernel::Float * RESTRICT input_data,
    typename MicroKernel::Float * RESTRICT output_data,
    const TensorIdx input_stride, const TensorIdx output_stride) const {
  this->ncont_tiler_(DualCounter<CONT, NCONT - 1>(), input_data, output_data,
      input_stride, output_stride);
  this->cont_tiler_(DualCounter<CONT - 1, NCONT - 1>(), input_data, output_data,
      input_stride, output_stride);
}


template <typename MicroKernel,
          TensorUInt CONT_LEN,
          TensorUInt NCONT_LEN>
template <TensorUInt CONT>
void MacroTrans<MicroKernel, CONT_LEN, NCONT_LEN>::ncont_tiler_(
    DualCounter<CONT, 0>,
    const typename MicroKernel::Float * RESTRICT input_data,
    typename MicroKernel::Float * RESTRICT output_data,
    const TensorIdx input_stride, const TensorIdx output_stride) const {
}


template <typename MicroKernel,
          TensorUInt CONT_LEN,
          TensorUInt NCONT_LEN>
template <TensorUInt CONT,
         TensorUInt NCONT>
void MacroTrans<MicroKernel, CONT_LEN, NCONT_LEN>::cont_tiler_(
    DualCounter<CONT, NCONT>,
    const typename MicroKernel::Float * RESTRICT input_data,
    typename MicroKernel::Float * RESTRICT output_data,
    const TensorIdx input_stride, const TensorIdx output_stride) const {
  this->cont_tiler_(DualCounter<CONT - 1, NCONT>(), input_data, output_data,
      input_stride, output_stride);
  this->kernel_.exec(input_data + CONT * this->kn_width_
          + NCONT * this->kn_width_ * input_stride,
      output_data + NCONT * this->kn_width_
          + CONT * this->kn_width_ * output_stride,
      input_stride, output_stride);
}


template <typename MicroKernel,
          TensorUInt CONT_LEN,
          TensorUInt NCONT_LEN>
template <TensorUInt NCONT>
void MacroTrans<MicroKernel, CONT_LEN, NCONT_LEN>::cont_tiler_(
    DualCounter<0, NCONT>,
    const typename MicroKernel::Float * RESTRICT input_data,
    typename MicroKernel::Float * RESTRICT output_data,
    const TensorIdx input_stride, const TensorIdx output_stride) const {
  this->kernel_.exec(input_data + NCONT * this->kn_width_ * input_stride,
      output_data + NCONT * this->kn_width_, input_stride, output_stride);
}


/*
 * Implementation for class MacroTransLinear
 */
template <typename FloatType>
void MacroTransLinear<FloatType>::set_coef(
    const DeducedFloatType<FloatType> alpha,
    const DeducedFloatType<FloatType> beta) {
  this->reg_alpha_ = alpha, this->reg_beta_ = beta;
}


template <typename FloatType>
void MacroTransLinear<FloatType>::exec(
    const FloatType * RESTRICT input_data, FloatType * RESTRICT output_data,
    const TensorIdx input_stride, const TensorIdx output_stride) const {
#pragma omp simd
  for (TensorIdx idx = 0; idx < input_stride; ++idx)
    output_data[idx] = this->reg_alpha_ * input_data[idx]
        + this->reg_beta_ * output_data[idx];
}


/*
 * Explicit template instantiation for class MacroTransLinear
 */
template class MacroTransLinear<float>;
template class MacroTransLinear<double>;
template class MacroTransLinear<FloatComplex>;
template class MacroTransLinear<DoubleComplex>;

/*
 * Explicit template instantiation for class MacroTrans
 */
template class MacroTrans<KernelTransFull<float>, 4, 4>;
template class MacroTrans<KernelTransFull<float>, 4, 3>;
template class MacroTrans<KernelTransFull<float>, 4, 2>;
template class MacroTrans<KernelTransFull<float>, 4, 1>;
template class MacroTrans<KernelTransFull<float>, 3, 4>;
template class MacroTrans<KernelTransFull<float>, 3, 3>;
template class MacroTrans<KernelTransFull<float>, 3, 2>;
template class MacroTrans<KernelTransFull<float>, 3, 1>;
template class MacroTrans<KernelTransFull<float>, 2, 4>;
template class MacroTrans<KernelTransFull<float>, 2, 3>;
template class MacroTrans<KernelTransFull<float>, 2, 2>;
template class MacroTrans<KernelTransFull<float>, 2, 1>;
template class MacroTrans<KernelTransFull<float>, 1, 4>;
template class MacroTrans<KernelTransFull<float>, 1, 3>;
template class MacroTrans<KernelTransFull<float>, 1, 2>;
template class MacroTrans<KernelTransFull<float>, 1, 1>;

template class MacroTrans<KernelTransFull<double>, 4, 4>;
template class MacroTrans<KernelTransFull<double>, 4, 3>;
template class MacroTrans<KernelTransFull<double>, 4, 2>;
template class MacroTrans<KernelTransFull<double>, 4, 1>;
template class MacroTrans<KernelTransFull<double>, 3, 4>;
template class MacroTrans<KernelTransFull<double>, 3, 3>;
template class MacroTrans<KernelTransFull<double>, 3, 2>;
template class MacroTrans<KernelTransFull<double>, 3, 1>;
template class MacroTrans<KernelTransFull<double>, 2, 4>;
template class MacroTrans<KernelTransFull<double>, 2, 3>;
template class MacroTrans<KernelTransFull<double>, 2, 2>;
template class MacroTrans<KernelTransFull<double>, 2, 1>;
template class MacroTrans<KernelTransFull<double>, 1, 4>;
template class MacroTrans<KernelTransFull<double>, 1, 3>;
template class MacroTrans<KernelTransFull<double>, 1, 2>;
template class MacroTrans<KernelTransFull<double>, 1, 1>;

template class MacroTrans<KernelTransFull<FloatComplex>, 4, 4>;
template class MacroTrans<KernelTransFull<FloatComplex>, 4, 3>;
template class MacroTrans<KernelTransFull<FloatComplex>, 4, 2>;
template class MacroTrans<KernelTransFull<FloatComplex>, 4, 1>;
template class MacroTrans<KernelTransFull<FloatComplex>, 3, 4>;
template class MacroTrans<KernelTransFull<FloatComplex>, 3, 3>;
template class MacroTrans<KernelTransFull<FloatComplex>, 3, 2>;
template class MacroTrans<KernelTransFull<FloatComplex>, 3, 1>;
template class MacroTrans<KernelTransFull<FloatComplex>, 2, 4>;
template class MacroTrans<KernelTransFull<FloatComplex>, 2, 3>;
template class MacroTrans<KernelTransFull<FloatComplex>, 2, 2>;
template class MacroTrans<KernelTransFull<FloatComplex>, 2, 1>;
template class MacroTrans<KernelTransFull<FloatComplex>, 1, 4>;
template class MacroTrans<KernelTransFull<FloatComplex>, 1, 3>;
template class MacroTrans<KernelTransFull<FloatComplex>, 1, 2>;
template class MacroTrans<KernelTransFull<FloatComplex>, 1, 1>;

template class MacroTrans<KernelTransFull<DoubleComplex>, 4, 4>;
template class MacroTrans<KernelTransFull<DoubleComplex>, 4, 3>;
template class MacroTrans<KernelTransFull<DoubleComplex>, 4, 2>;
template class MacroTrans<KernelTransFull<DoubleComplex>, 4, 1>;
template class MacroTrans<KernelTransFull<DoubleComplex>, 3, 4>;
template class MacroTrans<KernelTransFull<DoubleComplex>, 3, 3>;
template class MacroTrans<KernelTransFull<DoubleComplex>, 3, 2>;
template class MacroTrans<KernelTransFull<DoubleComplex>, 3, 1>;
template class MacroTrans<KernelTransFull<DoubleComplex>, 2, 4>;
template class MacroTrans<KernelTransFull<DoubleComplex>, 2, 3>;
template class MacroTrans<KernelTransFull<DoubleComplex>, 2, 2>;
template class MacroTrans<KernelTransFull<DoubleComplex>, 2, 1>;
template class MacroTrans<KernelTransFull<DoubleComplex>, 1, 4>;
template class MacroTrans<KernelTransFull<DoubleComplex>, 1, 3>;
template class MacroTrans<KernelTransFull<DoubleComplex>, 1, 2>;
template class MacroTrans<KernelTransFull<DoubleComplex>, 1, 1>;

template class MacroTrans<KernelTransHalf<float>, 4, 1>;
template class MacroTrans<KernelTransHalf<float>, 3, 1>;
template class MacroTrans<KernelTransHalf<float>, 2, 1>;
template class MacroTrans<KernelTransHalf<float>, 1, 4>;
template class MacroTrans<KernelTransHalf<float>, 1, 3>;
template class MacroTrans<KernelTransHalf<float>, 1, 2>;
template class MacroTrans<KernelTransHalf<float>, 1, 1>;

template class MacroTrans<KernelTransHalf<double>, 4, 1>;
template class MacroTrans<KernelTransHalf<double>, 3, 1>;
template class MacroTrans<KernelTransHalf<double>, 2, 1>;
template class MacroTrans<KernelTransHalf<double>, 1, 4>;
template class MacroTrans<KernelTransHalf<double>, 1, 3>;
template class MacroTrans<KernelTransHalf<double>, 1, 2>;
template class MacroTrans<KernelTransHalf<double>, 1, 1>;

template class MacroTrans<KernelTransHalf<FloatComplex>, 4, 1>;
template class MacroTrans<KernelTransHalf<FloatComplex>, 3, 1>;
template class MacroTrans<KernelTransHalf<FloatComplex>, 2, 1>;
template class MacroTrans<KernelTransHalf<FloatComplex>, 1, 4>;
template class MacroTrans<KernelTransHalf<FloatComplex>, 1, 3>;
template class MacroTrans<KernelTransHalf<FloatComplex>, 1, 2>;
template class MacroTrans<KernelTransHalf<FloatComplex>, 1, 1>;

template class MacroTrans<KernelTransHalf<DoubleComplex>, 4, 1>;
template class MacroTrans<KernelTransHalf<DoubleComplex>, 3, 1>;
template class MacroTrans<KernelTransHalf<DoubleComplex>, 2, 1>;
template class MacroTrans<KernelTransHalf<DoubleComplex>, 1, 4>;
template class MacroTrans<KernelTransHalf<DoubleComplex>, 1, 3>;
template class MacroTrans<KernelTransHalf<DoubleComplex>, 1, 2>;
template class MacroTrans<KernelTransHalf<DoubleComplex>, 1, 1>;

}
