#pragma once
#ifndef HPTC_KERNELS_MACRO_KERNEL_TRANS_TCC_
#define HPTC_KERNELS_MACRO_KERNEL_TRANS_TCC_

/*
 * Specialization and implementation for class MacroTransVec
 */
template <typename KernelFunc>
class MacroTransVec<KernelFunc, 0, 0> {
};


template <typename KernelFunc,
          GenNumType CONT_LEN,
          GenNumType NCONT_LEN>
typename KernelFunc::RegType
MacroTransVec<KernelFunc, CONT_LEN, NCONT_LEN>::reg_coef(
    const DeducedFloatType<typename KernelFunc::FLOAT> coef) {
  return KernelFunc::reg_coef(coef);
}


template <typename KernelFunc,
          GenNumType CONT_LEN,
          GenNumType NCONT_LEN>
constexpr GenNumType
MacroTransVec<KernelFunc, CONT_LEN, NCONT_LEN>::get_cont_len() const {
  return CONT_LEN * KernelFunc::kn_width;
}


template <typename KernelFunc,
          GenNumType CONT_LEN,
          GenNumType NCONT_LEN>
constexpr GenNumType
MacroTransVec<KernelFunc, CONT_LEN, NCONT_LEN>::get_ncont_len() const {
  return NCONT_LEN * KernelFunc::kn_width;
}


template <typename KernelFunc,
          GenNumType CONT_LEN,
          GenNumType NCONT_LEN>
void MacroTransVec<KernelFunc, CONT_LEN, NCONT_LEN>::operator()(
    const typename KernelFunc::FLOAT * RESTRICT input_data,
    typename KernelFunc::FLOAT * RESTRICT output_data,
    const TensorIdx input_stride, const TensorIdx output_stride,
    const typename KernelFunc::RegType &reg_alpha,
    const typename KernelFunc::RegType &reg_beta) const {
  constexpr auto kn_wd = KernelFunc::kn_width;
#pragma unroll_and_jam(CONT_LEN)
  for (GenNumType cont = 0; cont < CONT_LEN; ++cont)
#pragma unroll_and_jam(NCONT_LEN)
    for (GenNumType ncont = 0; ncont < NCONT_LEN; ++ncont)
      this->kernel_(input_data + cont * kn_wd + ncont * kn_wd * input_stride,
          output_data + ncont * kn_wd + cont * kn_wd * output_stride,
          input_stride, output_stride, reg_alpha, reg_beta);
}


/*
 * Explicit instantiation declaration for class MacroTransVec
 */
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_NONE>, 4, 4>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_NONE>, 4, 3>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_NONE>, 4, 2>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_NONE>, 4, 1>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_NONE>, 3, 4>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_NONE>, 3, 3>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_NONE>, 3, 2>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_NONE>, 3, 1>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_NONE>, 2, 4>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_NONE>, 2, 3>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_NONE>, 2, 2>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_NONE>, 2, 1>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_NONE>, 1, 4>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_NONE>, 1, 3>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_NONE>, 1, 2>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_NONE>, 1, 1>;

extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_ALPHA>, 4, 4>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_ALPHA>, 4, 3>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_ALPHA>, 4, 2>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_ALPHA>, 4, 1>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_ALPHA>, 3, 4>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_ALPHA>, 3, 3>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_ALPHA>, 3, 2>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_ALPHA>, 3, 1>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_ALPHA>, 2, 4>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_ALPHA>, 2, 3>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_ALPHA>, 2, 2>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_ALPHA>, 2, 1>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_ALPHA>, 1, 4>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_ALPHA>, 1, 3>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_ALPHA>, 1, 2>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_ALPHA>, 1, 1>;

extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_BETA>, 4, 4>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_BETA>, 4, 3>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_BETA>, 4, 2>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_BETA>, 4, 1>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_BETA>, 3, 4>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_BETA>, 3, 3>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_BETA>, 3, 2>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_BETA>, 3, 1>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_BETA>, 2, 4>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_BETA>, 2, 3>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_BETA>, 2, 2>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_BETA>, 2, 1>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_BETA>, 1, 4>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_BETA>, 1, 3>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_BETA>, 1, 2>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_BETA>, 1, 1>;

extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_BOTH>, 4, 4>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_BOTH>, 4, 3>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_BOTH>, 4, 2>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_BOTH>, 4, 1>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_BOTH>, 3, 4>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_BOTH>, 3, 3>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_BOTH>, 3, 2>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_BOTH>, 3, 1>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_BOTH>, 2, 4>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_BOTH>, 2, 3>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_BOTH>, 2, 2>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_BOTH>, 2, 1>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_BOTH>, 1, 4>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_BOTH>, 1, 3>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_BOTH>, 1, 2>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_BOTH>, 1, 1>;


extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_NONE>, 4, 4>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_NONE>, 4, 3>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_NONE>, 4, 2>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_NONE>, 4, 1>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_NONE>, 3, 4>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_NONE>, 3, 3>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_NONE>, 3, 2>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_NONE>, 3, 1>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_NONE>, 2, 4>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_NONE>, 2, 3>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_NONE>, 2, 2>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_NONE>, 2, 1>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_NONE>, 1, 4>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_NONE>, 1, 3>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_NONE>, 1, 2>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_NONE>, 1, 1>;

extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_ALPHA>, 4, 4>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_ALPHA>, 4, 3>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_ALPHA>, 4, 2>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_ALPHA>, 4, 1>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_ALPHA>, 3, 4>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_ALPHA>, 3, 3>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_ALPHA>, 3, 2>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_ALPHA>, 3, 1>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_ALPHA>, 2, 4>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_ALPHA>, 2, 3>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_ALPHA>, 2, 2>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_ALPHA>, 2, 1>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_ALPHA>, 1, 4>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_ALPHA>, 1, 3>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_ALPHA>, 1, 2>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_ALPHA>, 1, 1>;

extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_BETA>, 4, 4>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_BETA>, 4, 3>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_BETA>, 4, 2>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_BETA>, 4, 1>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_BETA>, 3, 4>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_BETA>, 3, 3>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_BETA>, 3, 2>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_BETA>, 3, 1>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_BETA>, 2, 4>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_BETA>, 2, 3>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_BETA>, 2, 2>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_BETA>, 2, 1>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_BETA>, 1, 4>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_BETA>, 1, 3>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_BETA>, 1, 2>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_BETA>, 1, 1>;

extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_BOTH>, 4, 4>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_BOTH>, 4, 3>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_BOTH>, 4, 2>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_BOTH>, 4, 1>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_BOTH>, 3, 4>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_BOTH>, 3, 3>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_BOTH>, 3, 2>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_BOTH>, 3, 1>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_BOTH>, 2, 4>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_BOTH>, 2, 3>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_BOTH>, 2, 2>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_BOTH>, 2, 1>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_BOTH>, 1, 4>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_BOTH>, 1, 3>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_BOTH>, 1, 2>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_BOTH>, 1, 1>;


extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_NONE>, 4, 4>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_NONE>, 4, 3>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_NONE>, 4, 2>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_NONE>, 4, 1>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_NONE>, 3, 4>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_NONE>, 3, 3>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_NONE>, 3, 2>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_NONE>, 3, 1>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_NONE>, 2, 4>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_NONE>, 2, 3>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_NONE>, 2, 2>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_NONE>, 2, 1>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_NONE>, 1, 4>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_NONE>, 1, 3>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_NONE>, 1, 2>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_NONE>, 1, 1>;

extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_ALPHA>, 4, 4>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_ALPHA>, 4, 3>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_ALPHA>, 4, 2>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_ALPHA>, 4, 1>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_ALPHA>, 3, 4>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_ALPHA>, 3, 3>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_ALPHA>, 3, 2>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_ALPHA>, 3, 1>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_ALPHA>, 2, 4>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_ALPHA>, 2, 3>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_ALPHA>, 2, 2>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_ALPHA>, 2, 1>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_ALPHA>, 1, 4>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_ALPHA>, 1, 3>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_ALPHA>, 1, 2>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_ALPHA>, 1, 1>;

extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_BETA>, 4, 4>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_BETA>, 4, 3>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_BETA>, 4, 2>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_BETA>, 4, 1>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_BETA>, 3, 4>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_BETA>, 3, 3>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_BETA>, 3, 2>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_BETA>, 3, 1>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_BETA>, 2, 4>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_BETA>, 2, 3>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_BETA>, 2, 2>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_BETA>, 2, 1>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_BETA>, 1, 4>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_BETA>, 1, 3>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_BETA>, 1, 2>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_BETA>, 1, 1>;

extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_BOTH>, 4, 4>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_BOTH>, 4, 3>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_BOTH>, 4, 2>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_BOTH>, 4, 1>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_BOTH>, 3, 4>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_BOTH>, 3, 3>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_BOTH>, 3, 2>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_BOTH>, 3, 1>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_BOTH>, 2, 4>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_BOTH>, 2, 3>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_BOTH>, 2, 2>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_BOTH>, 2, 1>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_BOTH>, 1, 4>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_BOTH>, 1, 3>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_BOTH>, 1, 2>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_BOTH>, 1, 1>;


extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_NONE>, 4, 4>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_NONE>, 4, 3>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_NONE>, 4, 2>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_NONE>, 4, 1>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_NONE>, 3, 4>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_NONE>, 3, 3>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_NONE>, 3, 2>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_NONE>, 3, 1>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_NONE>, 2, 4>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_NONE>, 2, 3>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_NONE>, 2, 2>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_NONE>, 2, 1>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_NONE>, 1, 4>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_NONE>, 1, 3>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_NONE>, 1, 2>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_NONE>, 1, 1>;

extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_ALPHA>, 4, 4>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_ALPHA>, 4, 3>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_ALPHA>, 4, 2>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_ALPHA>, 4, 1>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_ALPHA>, 3, 4>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_ALPHA>, 3, 3>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_ALPHA>, 3, 2>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_ALPHA>, 3, 1>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_ALPHA>, 2, 4>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_ALPHA>, 2, 3>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_ALPHA>, 2, 2>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_ALPHA>, 2, 1>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_ALPHA>, 1, 4>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_ALPHA>, 1, 3>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_ALPHA>, 1, 2>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_ALPHA>, 1, 1>;

extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_BETA>, 4, 4>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_BETA>, 4, 3>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_BETA>, 4, 2>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_BETA>, 4, 1>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_BETA>, 3, 4>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_BETA>, 3, 3>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_BETA>, 3, 2>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_BETA>, 3, 1>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_BETA>, 2, 4>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_BETA>, 2, 3>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_BETA>, 2, 2>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_BETA>, 2, 1>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_BETA>, 1, 4>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_BETA>, 1, 3>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_BETA>, 1, 2>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_BETA>, 1, 1>;

extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_BOTH>, 4, 4>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_BOTH>, 4, 3>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_BOTH>, 4, 2>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_BOTH>, 4, 1>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_BOTH>, 3, 4>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_BOTH>, 3, 3>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_BOTH>, 3, 2>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_BOTH>, 3, 1>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_BOTH>, 2, 4>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_BOTH>, 2, 3>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_BOTH>, 2, 2>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_BOTH>, 2, 1>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_BOTH>, 1, 4>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_BOTH>, 1, 3>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_BOTH>, 1, 2>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_BOTH>, 1, 1>;


extern template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_NONE>, 4, 1>;
extern template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_NONE>, 3, 1>;
extern template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_NONE>, 2, 1>;
extern template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_NONE>, 1, 4>;
extern template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_NONE>, 1, 3>;
extern template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_NONE>, 1, 2>;
extern template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_NONE>, 1, 1>;

extern template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_ALPHA>, 4, 1>;
extern template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_ALPHA>, 3, 1>;
extern template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_ALPHA>, 2, 1>;
extern template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_ALPHA>, 1, 4>;
extern template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_ALPHA>, 1, 3>;
extern template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_ALPHA>, 1, 2>;
extern template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_ALPHA>, 1, 1>;

extern template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_BETA>, 4, 1>;
extern template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_BETA>, 3, 1>;
extern template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_BETA>, 2, 1>;
extern template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_BETA>, 1, 4>;
extern template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_BETA>, 1, 3>;
extern template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_BETA>, 1, 2>;
extern template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_BETA>, 1, 1>;

extern template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_BOTH>, 4, 1>;
extern template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_BOTH>, 3, 1>;
extern template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_BOTH>, 2, 1>;
extern template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_BOTH>, 1, 4>;
extern template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_BOTH>, 1, 3>;
extern template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_BOTH>, 1, 2>;
extern template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_BOTH>, 1, 1>;


extern template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_NONE>, 4, 1>;
extern template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_NONE>, 3, 1>;
extern template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_NONE>, 2, 1>;
extern template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_NONE>, 1, 4>;
extern template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_NONE>, 1, 3>;
extern template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_NONE>, 1, 2>;
extern template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_NONE>, 1, 1>;

extern template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_ALPHA>, 4, 1>;
extern template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_ALPHA>, 3, 1>;
extern template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_ALPHA>, 2, 1>;
extern template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_ALPHA>, 1, 4>;
extern template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_ALPHA>, 1, 3>;
extern template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_ALPHA>, 1, 2>;
extern template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_ALPHA>, 1, 1>;

extern template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_BETA>, 4, 1>;
extern template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_BETA>, 3, 1>;
extern template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_BETA>, 2, 1>;
extern template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_BETA>, 1, 4>;
extern template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_BETA>, 1, 3>;
extern template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_BETA>, 1, 2>;
extern template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_BETA>, 1, 1>;

extern template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_BOTH>, 4, 1>;
extern template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_BOTH>, 3, 1>;
extern template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_BOTH>, 2, 1>;
extern template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_BOTH>, 1, 4>;
extern template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_BOTH>, 1, 3>;
extern template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_BOTH>, 1, 2>;
extern template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_BOTH>, 1, 1>;


extern template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_NONE>, 4, 1>;
extern template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_NONE>, 3, 1>;
extern template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_NONE>, 2, 1>;
extern template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_NONE>, 1, 4>;
extern template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_NONE>, 1, 3>;
extern template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_NONE>, 1, 2>;
extern template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_NONE>, 1, 1>;

extern template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_ALPHA>, 4, 1>;
extern template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_ALPHA>, 3, 1>;
extern template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_ALPHA>, 2, 1>;
extern template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_ALPHA>, 1, 4>;
extern template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_ALPHA>, 1, 3>;
extern template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_ALPHA>, 1, 2>;
extern template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_ALPHA>, 1, 1>;

extern template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_BETA>, 4, 1>;
extern template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_BETA>, 3, 1>;
extern template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_BETA>, 2, 1>;
extern template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_BETA>, 1, 4>;
extern template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_BETA>, 1, 3>;
extern template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_BETA>, 1, 2>;
extern template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_BETA>, 1, 1>;

extern template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_BOTH>, 4, 1>;
extern template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_BOTH>, 3, 1>;
extern template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_BOTH>, 2, 1>;
extern template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_BOTH>, 1, 4>;
extern template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_BOTH>, 1, 3>;
extern template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_BOTH>, 1, 2>;
extern template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_BOTH>, 1, 1>;


extern template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_NONE>, 4, 1>;
extern template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_NONE>, 3, 1>;
extern template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_NONE>, 2, 1>;
extern template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_NONE>, 1, 4>;
extern template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_NONE>, 1, 3>;
extern template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_NONE>, 1, 2>;
extern template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_NONE>, 1, 1>;

extern template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_ALPHA>, 4, 1>;
extern template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_ALPHA>, 3, 1>;
extern template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_ALPHA>, 2, 1>;
extern template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_ALPHA>, 1, 4>;
extern template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_ALPHA>, 1, 3>;
extern template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_ALPHA>, 1, 2>;
extern template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_ALPHA>, 1, 1>;

extern template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_BETA>, 4, 1>;
extern template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_BETA>, 3, 1>;
extern template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_BETA>, 2, 1>;
extern template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_BETA>, 1, 4>;
extern template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_BETA>, 1, 3>;
extern template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_BETA>, 1, 2>;
extern template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_BETA>, 1, 1>;

extern template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_BOTH>, 4, 1>;
extern template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_BOTH>, 3, 1>;
extern template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_BOTH>, 2, 1>;
extern template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_BOTH>, 1, 4>;
extern template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_BOTH>, 1, 3>;
extern template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_BOTH>, 1, 2>;
extern template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_BOTH>, 1, 1>;


/*
 * Avoid template instantiation for class MacroTransLinear
 */
extern template class MacroTransLinear<float, CoefUsageTrans::USE_NONE>;
extern template class MacroTransLinear<float, CoefUsageTrans::USE_ALPHA>;
extern template class MacroTransLinear<float, CoefUsageTrans::USE_BETA>;
extern template class MacroTransLinear<float, CoefUsageTrans::USE_BOTH>;
extern template class MacroTransLinear<double, CoefUsageTrans::USE_NONE>;
extern template class MacroTransLinear<double, CoefUsageTrans::USE_ALPHA>;
extern template class MacroTransLinear<double, CoefUsageTrans::USE_BETA>;
extern template class MacroTransLinear<double, CoefUsageTrans::USE_BOTH>;
extern template class MacroTransLinear<FloatComplex, CoefUsageTrans::USE_NONE>;
extern template class MacroTransLinear<FloatComplex, CoefUsageTrans::USE_ALPHA>;
extern template class MacroTransLinear<FloatComplex, CoefUsageTrans::USE_BETA>;
extern template class MacroTransLinear<FloatComplex, CoefUsageTrans::USE_BOTH>;
extern template class MacroTransLinear<DoubleComplex, CoefUsageTrans::USE_NONE>;
extern template class MacroTransLinear<DoubleComplex,
    CoefUsageTrans::USE_ALPHA>;
extern template class MacroTransLinear<DoubleComplex, CoefUsageTrans::USE_BETA>;
extern template class MacroTransLinear<DoubleComplex, CoefUsageTrans::USE_BOTH>;

#endif // HPTC_KERNELS_MACRO_KERNEL_TRANS_TCC_
