#include <hptc/kernels/macro_kernel_trans.h>

#include <algorithm>

#include <hptc/types.h>
#include <hptc/util/util_trans.h>


namespace hptc {

/*
 * Implementation for class MacroTransLinear
 */
template <typename FloatType,
          CoefUsageTrans USAGE>
void MacroTransLinear<FloatType, USAGE>::set_coef(
    const DeducedFloatType<FloatType> alpha,
    const DeducedFloatType<FloatType> beta) {
  this->reg_alpha_ = alpha, this->reg_beta_ = beta;
}


template <typename FloatType,
          CoefUsageTrans USAGE>
void MacroTransLinear<FloatType, USAGE>::exec(
    const FloatType * RESTRICT input_data, FloatType * RESTRICT output_data,
    const TensorIdx input_stride, const TensorIdx output_stride) const {
  if (USAGE == CoefUsageTrans::USE_NONE)
    std::copy(input_data, input_data + input_stride, output_data);
  else if (USAGE == CoefUsageTrans::USE_ALPHA)
#pragma omp simd
    for (TensorIdx idx = 0; idx < input_stride; ++idx)
      output_data[idx] = this->reg_alpha_ * input_data[idx];
  else if (USAGE == CoefUsageTrans::USE_BETA)
#pragma omp simd
    for (TensorIdx idx = 0; idx < input_stride; ++idx)
      output_data[idx] = input_data[idx] + this->reg_beta_ * output_data[idx];
  else
#pragma omp simd
    for (TensorIdx idx = 0; idx < input_stride; ++idx)
      output_data[idx] = this->reg_alpha_ * input_data[idx]
          + this->reg_beta_ * output_data[idx];
}


/*
 * Explicit template instantiation for class MacroTransLinear
 */
template class MacroTransLinear<float, CoefUsageTrans::USE_NONE>;
template class MacroTransLinear<double, CoefUsageTrans::USE_NONE>;
template class MacroTransLinear<FloatComplex, CoefUsageTrans::USE_NONE>;
template class MacroTransLinear<DoubleComplex, CoefUsageTrans::USE_NONE>;
template class MacroTransLinear<float, CoefUsageTrans::USE_ALPHA>;
template class MacroTransLinear<double, CoefUsageTrans::USE_ALPHA>;
template class MacroTransLinear<FloatComplex, CoefUsageTrans::USE_ALPHA>;
template class MacroTransLinear<DoubleComplex, CoefUsageTrans::USE_ALPHA>;
template class MacroTransLinear<float, CoefUsageTrans::USE_BETA>;
template class MacroTransLinear<double, CoefUsageTrans::USE_BETA>;
template class MacroTransLinear<FloatComplex, CoefUsageTrans::USE_BETA>;
template class MacroTransLinear<DoubleComplex, CoefUsageTrans::USE_BETA>;
template class MacroTransLinear<float, CoefUsageTrans::USE_BOTH>;
template class MacroTransLinear<double, CoefUsageTrans::USE_BOTH>;
template class MacroTransLinear<FloatComplex, CoefUsageTrans::USE_BOTH>;
template class MacroTransLinear<DoubleComplex, CoefUsageTrans::USE_BOTH>;

}
