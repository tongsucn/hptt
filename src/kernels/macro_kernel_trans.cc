#include <hptc/kernels/macro_kernel_trans.h>

#include <algorithm>

#include <hptc/types.h>
#include <hptc/util/util_trans.h>


#include <iostream>

namespace hptc {

/*
 * Implementation for class MacroTransLinear
 */
template <typename FloatType,
          CoefUsageTrans USAGE>
DeducedRegType<FloatType, KernelTypeTrans::KERNEL_LINE>
MacroTransLinear<FloatType, USAGE>::reg_coef(
    const DeducedFloatType<FloatType> coef) {
  return coef;
}


template <typename FloatType,
          CoefUsageTrans USAGE>
void MacroTransLinear<FloatType, USAGE>::operator()(
    const FloatType * RESTRICT input_data, FloatType * RESTRICT output_data,
    const TensorIdx input_stride, const TensorIdx output_stride,
    const RegType &alpha, const RegType &beta) const {
  if (USAGE == CoefUsageTrans::USE_NONE)
    std::copy(input_data, input_data + input_stride, output_data);
  else if (USAGE == CoefUsageTrans::USE_ALPHA)
#pragma simd vectorlengthfor(FloatType)
    for (TensorIdx idx = 0; idx < input_stride; ++idx)
      output_data[idx] = alpha * input_data[idx];
  else if (USAGE == CoefUsageTrans::USE_BETA)
#pragma simd vectorlengthfor(FloatType)
    for (TensorIdx idx = 0; idx < input_stride; ++idx)
      output_data[idx] = input_data[idx] + beta * output_data[idx];
  else
#pragma simd vectorlengthfor(FloatType)
    for (TensorIdx idx = 0; idx < input_stride; ++idx)
      output_data[idx] = alpha * input_data[idx] + beta * output_data[idx];
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
