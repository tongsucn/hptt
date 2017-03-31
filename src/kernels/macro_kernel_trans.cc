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
      kn_width_(MicroKernel::KN_WIDTH) {
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
TensorUInt MacroTrans<MicroKernel, CONT_LEN, NCONT_LEN>::get_cont_len() const {
  return CONT_LEN *this->kn_width_;
}


template <typename MicroKernel,
          TensorUInt CONT_LEN,
          TensorUInt NCONT_LEN>
TensorUInt MacroTrans<MicroKernel, CONT_LEN, NCONT_LEN>::get_ncont_len() const {
  return NCONT_LEN *this->kn_width_;
}


template <typename MicroKernel,
          TensorUInt CONT_LEN,
          TensorUInt NCONT_LEN>
void MacroTrans<MicroKernel, CONT_LEN, NCONT_LEN>::exec(
    const typename MicroKernel::Float *in_data,
    typename MicroKernel::Float *out_data, const TensorIdx input_stride,
    const TensorIdx output_stride) const {
  // Tiling kernels in non-continuous direction
  this->cont_tiler_(DualCounter<CONT_LEN, NCONT_LEN>(), in_data,
      out_data, input_stride, output_stride);
}


template <typename MicroKernel,
          TensorUInt CONT_LEN,
          TensorUInt NCONT_LEN>
template <TensorUInt CONT,
         TensorUInt NCONT>
void MacroTrans<MicroKernel, CONT_LEN, NCONT_LEN>::cont_tiler_(
    DualCounter<CONT, NCONT>, const typename MicroKernel::Float *in_data,
    typename MicroKernel::Float *out_data, const TensorIdx input_stride,
    const TensorIdx output_stride) const {
  this->cont_tiler_(DualCounter<CONT - 1, NCONT>(), in_data, out_data,
      input_stride, output_stride);
  this->ncont_tiler_(DualCounter<CONT - 1, NCONT - 1>(), in_data, out_data,
      input_stride, output_stride);
}


template <typename MicroKernel,
          TensorUInt CONT_LEN,
          TensorUInt NCONT_LEN>
template <TensorUInt NCONT>
void MacroTrans<MicroKernel, CONT_LEN, NCONT_LEN>::cont_tiler_(
    DualCounter<0, NCONT>, const typename MicroKernel::Float *in_data,
    typename MicroKernel::Float *out_data, const TensorIdx input_stride,
    const TensorIdx output_stride) const {
}


template <typename MicroKernel,
          TensorUInt CONT_LEN,
          TensorUInt NCONT_LEN>
template <TensorUInt CONT,
         TensorUInt NCONT>
void MacroTrans<MicroKernel, CONT_LEN, NCONT_LEN>::ncont_tiler_(
    DualCounter<CONT, NCONT>, const typename MicroKernel::Float *in_data,
    typename MicroKernel::Float *out_data, const TensorIdx input_stride,
    const TensorIdx output_stride) const {
  this->ncont_tiler_(DualCounter<CONT, NCONT - 1>(), in_data, out_data,
      input_stride, output_stride);
  this->kernel_.exec(in_data + CONT * this->kn_width_
          + NCONT * this->kn_width_ *input_stride,
      out_data + NCONT *this->kn_width_
          + CONT * this->kn_width_ *output_stride,
      input_stride, output_stride);
}


template <typename MicroKernel,
          TensorUInt CONT_LEN,
          TensorUInt NCONT_LEN>
template <TensorUInt CONT>
void MacroTrans<MicroKernel, CONT_LEN, NCONT_LEN>::ncont_tiler_(
    DualCounter<CONT, 0>, const typename MicroKernel::Float *in_data,
    typename MicroKernel::Float *out_data, const TensorIdx input_stride,
    const TensorIdx output_stride) const {
  this->kernel_.exec(in_data + CONT * this->kn_width_,
      out_data + CONT * this->kn_width_ *output_stride, input_stride,
      output_stride);
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
void MacroTransLinear<FloatType>::exec(const FloatType *in_data,
    FloatType *out_data, const TensorIdx input_stride,
    const TensorIdx output_stride) const {
  for (TensorIdx idx = 0; idx < input_stride; ++idx)
    out_data[idx] = this->reg_alpha_ * in_data[idx]
        + this->reg_beta_ * out_data[idx];
}


/*
 * Implementation for class MacroTransScalar
 */
template <typename FloatType>
void MacroTransScalar<FloatType>::set_coef(
    const DeducedFloatType<FloatType> alpha,
    const DeducedFloatType<FloatType> beta) {
  this->alpha_ = alpha, this->beta_ = beta;
}


template <typename FloatType>
void MacroTransScalar<FloatType>::exec(const FloatType *in_data,
    FloatType *out_data, const TensorIdx input_size,
    const TensorIdx output_size) const {
  for (TensorIdx idx = 0; idx < input_size; ++idx)
    out_data[idx] = this->alpha_ * in_data[idx] + this->beta_ * out_data[idx];

  // Zero padding rest
  using Deduced = DeducedFloatType<FloatType>;
  auto underlying_ptr = reinterpret_cast<Deduced *>(out_data);
  constexpr TensorUInt inner_size = sizeof(FloatType) / sizeof(Deduced);
  std::fill(underlying_ptr + inner_size * input_size,
      underlying_ptr + inner_size * output_size, static_cast<Deduced>(0.0));
}


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

/*
 * Explicit template instantiation for class MacroTransLinear
 */
template class MacroTransLinear<float>;
template class MacroTransLinear<double>;
template class MacroTransLinear<FloatComplex>;
template class MacroTransLinear<DoubleComplex>;

/*
 * Explicit template instantiation for class MacroTransScalar
 */
template class MacroTransScalar<float>;
template class MacroTransScalar<double>;
template class MacroTransScalar<FloatComplex>;
template class MacroTransScalar<DoubleComplex>;

}
