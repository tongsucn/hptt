#pragma once
#ifndef HPTC_KERNELS_MACRO_KERNEL_TRANS_TCC_
#define HPTC_KERNELS_MACRO_KERNEL_TRANS_TCC_

/*
 * Avoid template instantiation for class MacroTransVec
 */
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_NONE>, 4, 4>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_ALPHA>, 4, 4>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_BETA>, 4, 4>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_BOTH>, 4, 4>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_NONE>, 4, 4>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_ALPHA>, 4, 4>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_BETA>, 4, 4>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_BOTH>, 4, 4>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_NONE>, 4, 4>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_ALPHA>, 4, 4>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_BETA>, 4, 4>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_BOTH>, 4, 4>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_NONE>, 4, 4>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_ALPHA>, 4, 4>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_BETA>, 4, 4>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_BOTH>, 4, 4>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_NONE>, 1, 4>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_ALPHA>, 1, 4>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_BETA>, 1, 4>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_BOTH>, 1, 4>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_NONE>, 1, 4>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_ALPHA>, 1, 4>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_BETA>, 1, 4>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_BOTH>, 1, 4>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_NONE>, 1, 4>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_ALPHA>, 1, 4>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_BETA>, 1, 4>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_BOTH>, 1, 4>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_NONE>, 1, 4>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_ALPHA>, 1, 4>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_BETA>, 1, 4>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_BOTH>, 1, 4>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_NONE>, 4, 1>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_ALPHA>, 4, 1>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_BETA>, 4, 1>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_BOTH>, 4, 1>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_NONE>, 4, 1>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_ALPHA>, 4, 1>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_BETA>, 4, 1>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_BOTH>, 4, 1>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_NONE>, 4, 1>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_ALPHA>, 4, 1>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_BETA>, 4, 1>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_BOTH>, 4, 1>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_NONE>, 4, 1>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_ALPHA>, 4, 1>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_BETA>, 4, 1>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_BOTH>, 4, 1>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_NONE>, 1, 1>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_ALPHA>, 1, 1>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_BETA>, 1, 1>;
extern template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_BOTH>, 1, 1>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_NONE>, 1, 1>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_ALPHA>, 1, 1>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_BETA>, 1, 1>;
extern template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_BOTH>, 1, 1>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_NONE>, 1, 1>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_ALPHA>, 1, 1>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_BETA>, 1, 1>;
extern template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_BOTH>, 1, 1>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_NONE>, 1, 1>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_ALPHA>, 1, 1>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_BETA>, 1, 1>;
extern template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_BOTH>, 1, 1>;

extern template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_NONE>, 1, 1>;
extern template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_ALPHA>, 1, 1>;
extern template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_BETA>, 1, 1>;
extern template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_BOTH>, 1, 1>;
extern template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_NONE>, 1, 1>;
extern template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_ALPHA>, 1, 1>;
extern template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_BETA>, 1, 1>;
extern template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_BOTH>, 1, 1>;
extern template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_NONE>, 1, 1>;
extern template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_ALPHA>, 1, 1>;
extern template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_BETA>, 1, 1>;
extern template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_BOTH>, 1, 1>;
extern template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_NONE>, 1, 1>;
extern template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_ALPHA>, 1, 1>;
extern template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_BETA>, 1, 1>;
extern template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_BOTH>, 1, 1>;
extern template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_NONE>, 1, 2>;
extern template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_ALPHA>, 1, 2>;
extern template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_BETA>, 1, 2>;
extern template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_BOTH>, 1, 2>;
extern template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_NONE>, 1, 2>;
extern template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_ALPHA>, 1, 2>;
extern template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_BETA>, 1, 2>;
extern template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_BOTH>, 1, 2>;
extern template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_NONE>, 1, 2>;
extern template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_ALPHA>, 1, 2>;
extern template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_BETA>, 1, 2>;
extern template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_BOTH>, 1, 2>;
extern template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_NONE>, 1, 2>;
extern template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_ALPHA>, 1, 2>;
extern template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_BETA>, 1, 2>;
extern template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_BOTH>, 1, 2>;
extern template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_NONE>, 2, 1>;
extern template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_ALPHA>, 2, 1>;
extern template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_BETA>, 2, 1>;
extern template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_BOTH>, 2, 1>;
extern template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_NONE>, 2, 1>;
extern template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_ALPHA>, 2, 1>;
extern template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_BETA>, 2, 1>;
extern template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_BOTH>, 2, 1>;
extern template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_NONE>, 2, 1>;
extern template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_ALPHA>, 2, 1>;
extern template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_BETA>, 2, 1>;
extern template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_BOTH>, 2, 1>;
extern template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_NONE>, 2, 1>;
extern template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_ALPHA>, 2, 1>;
extern template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_BETA>, 2, 1>;
extern template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_BOTH>, 2, 1>;

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
