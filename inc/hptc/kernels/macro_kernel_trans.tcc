#pragma once
#ifndef HPTC_KERNELS_MACRO_KERNEL_TRANS_TCC_
#define HPTC_KERNELS_MACRO_KERNEL_TRANS_TCC_

/*
 * Explicit instantiation declaration for class MacroTrans
 */
extern template class MacroTrans<KernelTransFull<float>, 4, 4>;
extern template class MacroTrans<KernelTransFull<float>, 4, 3>;
extern template class MacroTrans<KernelTransFull<float>, 4, 2>;
extern template class MacroTrans<KernelTransFull<float>, 4, 1>;
extern template class MacroTrans<KernelTransFull<float>, 3, 4>;
extern template class MacroTrans<KernelTransFull<float>, 3, 3>;
extern template class MacroTrans<KernelTransFull<float>, 3, 2>;
extern template class MacroTrans<KernelTransFull<float>, 3, 1>;
extern template class MacroTrans<KernelTransFull<float>, 2, 4>;
extern template class MacroTrans<KernelTransFull<float>, 2, 3>;
extern template class MacroTrans<KernelTransFull<float>, 2, 2>;
extern template class MacroTrans<KernelTransFull<float>, 2, 1>;
extern template class MacroTrans<KernelTransFull<float>, 1, 4>;
extern template class MacroTrans<KernelTransFull<float>, 1, 3>;
extern template class MacroTrans<KernelTransFull<float>, 1, 2>;
extern template class MacroTrans<KernelTransFull<float>, 1, 1>;

extern template class MacroTrans<KernelTransFull<double>, 4, 4>;
extern template class MacroTrans<KernelTransFull<double>, 4, 3>;
extern template class MacroTrans<KernelTransFull<double>, 4, 2>;
extern template class MacroTrans<KernelTransFull<double>, 4, 1>;
extern template class MacroTrans<KernelTransFull<double>, 3, 4>;
extern template class MacroTrans<KernelTransFull<double>, 3, 3>;
extern template class MacroTrans<KernelTransFull<double>, 3, 2>;
extern template class MacroTrans<KernelTransFull<double>, 3, 1>;
extern template class MacroTrans<KernelTransFull<double>, 2, 4>;
extern template class MacroTrans<KernelTransFull<double>, 2, 3>;
extern template class MacroTrans<KernelTransFull<double>, 2, 2>;
extern template class MacroTrans<KernelTransFull<double>, 2, 1>;
extern template class MacroTrans<KernelTransFull<double>, 1, 4>;
extern template class MacroTrans<KernelTransFull<double>, 1, 3>;
extern template class MacroTrans<KernelTransFull<double>, 1, 2>;
extern template class MacroTrans<KernelTransFull<double>, 1, 1>;

extern template class MacroTrans<KernelTransFull<FloatComplex>, 4, 4>;
extern template class MacroTrans<KernelTransFull<FloatComplex>, 4, 3>;
extern template class MacroTrans<KernelTransFull<FloatComplex>, 4, 2>;
extern template class MacroTrans<KernelTransFull<FloatComplex>, 4, 1>;
extern template class MacroTrans<KernelTransFull<FloatComplex>, 3, 4>;
extern template class MacroTrans<KernelTransFull<FloatComplex>, 3, 3>;
extern template class MacroTrans<KernelTransFull<FloatComplex>, 3, 2>;
extern template class MacroTrans<KernelTransFull<FloatComplex>, 3, 1>;
extern template class MacroTrans<KernelTransFull<FloatComplex>, 2, 4>;
extern template class MacroTrans<KernelTransFull<FloatComplex>, 2, 3>;
extern template class MacroTrans<KernelTransFull<FloatComplex>, 2, 2>;
extern template class MacroTrans<KernelTransFull<FloatComplex>, 2, 1>;
extern template class MacroTrans<KernelTransFull<FloatComplex>, 1, 4>;
extern template class MacroTrans<KernelTransFull<FloatComplex>, 1, 3>;
extern template class MacroTrans<KernelTransFull<FloatComplex>, 1, 2>;
extern template class MacroTrans<KernelTransFull<FloatComplex>, 1, 1>;

extern template class MacroTrans<KernelTransFull<DoubleComplex>, 4, 4>;
extern template class MacroTrans<KernelTransFull<DoubleComplex>, 4, 3>;
extern template class MacroTrans<KernelTransFull<DoubleComplex>, 4, 2>;
extern template class MacroTrans<KernelTransFull<DoubleComplex>, 4, 1>;
extern template class MacroTrans<KernelTransFull<DoubleComplex>, 3, 4>;
extern template class MacroTrans<KernelTransFull<DoubleComplex>, 3, 3>;
extern template class MacroTrans<KernelTransFull<DoubleComplex>, 3, 2>;
extern template class MacroTrans<KernelTransFull<DoubleComplex>, 3, 1>;
extern template class MacroTrans<KernelTransFull<DoubleComplex>, 2, 4>;
extern template class MacroTrans<KernelTransFull<DoubleComplex>, 2, 3>;
extern template class MacroTrans<KernelTransFull<DoubleComplex>, 2, 2>;
extern template class MacroTrans<KernelTransFull<DoubleComplex>, 2, 1>;
extern template class MacroTrans<KernelTransFull<DoubleComplex>, 1, 4>;
extern template class MacroTrans<KernelTransFull<DoubleComplex>, 1, 3>;
extern template class MacroTrans<KernelTransFull<DoubleComplex>, 1, 2>;
extern template class MacroTrans<KernelTransFull<DoubleComplex>, 1, 1>;

extern template class MacroTrans<KernelTransHalf<float>, 4, 1>;
extern template class MacroTrans<KernelTransHalf<float>, 3, 1>;
extern template class MacroTrans<KernelTransHalf<float>, 2, 1>;
extern template class MacroTrans<KernelTransHalf<float>, 1, 4>;
extern template class MacroTrans<KernelTransHalf<float>, 1, 3>;
extern template class MacroTrans<KernelTransHalf<float>, 1, 2>;
extern template class MacroTrans<KernelTransHalf<float>, 1, 1>;

extern template class MacroTrans<KernelTransHalf<double>, 4, 1>;
extern template class MacroTrans<KernelTransHalf<double>, 3, 1>;
extern template class MacroTrans<KernelTransHalf<double>, 2, 1>;
extern template class MacroTrans<KernelTransHalf<double>, 1, 4>;
extern template class MacroTrans<KernelTransHalf<double>, 1, 3>;
extern template class MacroTrans<KernelTransHalf<double>, 1, 2>;
extern template class MacroTrans<KernelTransHalf<double>, 1, 1>;

extern template class MacroTrans<KernelTransHalf<FloatComplex>, 4, 1>;
extern template class MacroTrans<KernelTransHalf<FloatComplex>, 3, 1>;
extern template class MacroTrans<KernelTransHalf<FloatComplex>, 2, 1>;
extern template class MacroTrans<KernelTransHalf<FloatComplex>, 1, 4>;
extern template class MacroTrans<KernelTransHalf<FloatComplex>, 1, 3>;
extern template class MacroTrans<KernelTransHalf<FloatComplex>, 1, 2>;
extern template class MacroTrans<KernelTransHalf<FloatComplex>, 1, 1>;

extern template class MacroTrans<KernelTransHalf<DoubleComplex>, 4, 1>;
extern template class MacroTrans<KernelTransHalf<DoubleComplex>, 3, 1>;
extern template class MacroTrans<KernelTransHalf<DoubleComplex>, 2, 1>;
extern template class MacroTrans<KernelTransHalf<DoubleComplex>, 1, 4>;
extern template class MacroTrans<KernelTransHalf<DoubleComplex>, 1, 3>;
extern template class MacroTrans<KernelTransHalf<DoubleComplex>, 1, 2>;
extern template class MacroTrans<KernelTransHalf<DoubleComplex>, 1, 1>;


/*
 * Explicit template instantiation declaration for class MacroTransLinear
 */
extern template class MacroTransLinear<float>;
extern template class MacroTransLinear<double>;
extern template class MacroTransLinear<FloatComplex>;
extern template class MacroTransLinear<DoubleComplex>;


/*
 * Explicit template instantiation declaration for class MacroTransScalar
 */
extern template class MacroTransScalar<float>;
extern template class MacroTransScalar<double>;
extern template class MacroTransScalar<FloatComplex>;
extern template class MacroTransScalar<DoubleComplex>;

#endif // HPTC_KERNELS_MACRO_KERNEL_TRANS_TCC_
