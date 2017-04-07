#pragma once
#ifndef HPTT_KERNELS_MACRO_KERNEL_TRANS_TCC_
#define HPTT_KERNELS_MACRO_KERNEL_TRANS_TCC_

/*
 * Explicit instantiation declaration for class MacroTrans
 */
extern template class MacroTrans<KernelTransFull<float, true>, 4, 4>;
extern template class MacroTrans<KernelTransFull<float, true>, 4, 3>;
extern template class MacroTrans<KernelTransFull<float, true>, 4, 2>;
extern template class MacroTrans<KernelTransFull<float, true>, 4, 1>;
extern template class MacroTrans<KernelTransFull<float, true>, 3, 4>;
extern template class MacroTrans<KernelTransFull<float, true>, 3, 3>;
extern template class MacroTrans<KernelTransFull<float, true>, 3, 2>;
extern template class MacroTrans<KernelTransFull<float, true>, 3, 1>;
extern template class MacroTrans<KernelTransFull<float, true>, 2, 4>;
extern template class MacroTrans<KernelTransFull<float, true>, 2, 3>;
extern template class MacroTrans<KernelTransFull<float, true>, 2, 2>;
extern template class MacroTrans<KernelTransFull<float, true>, 2, 1>;
extern template class MacroTrans<KernelTransFull<float, true>, 1, 4>;
extern template class MacroTrans<KernelTransFull<float, true>, 1, 3>;
extern template class MacroTrans<KernelTransFull<float, true>, 1, 2>;
extern template class MacroTrans<KernelTransFull<float, true>, 1, 1>;

extern template class MacroTrans<KernelTransFull<double, true>, 4, 4>;
extern template class MacroTrans<KernelTransFull<double, true>, 4, 3>;
extern template class MacroTrans<KernelTransFull<double, true>, 4, 2>;
extern template class MacroTrans<KernelTransFull<double, true>, 4, 1>;
extern template class MacroTrans<KernelTransFull<double, true>, 3, 4>;
extern template class MacroTrans<KernelTransFull<double, true>, 3, 3>;
extern template class MacroTrans<KernelTransFull<double, true>, 3, 2>;
extern template class MacroTrans<KernelTransFull<double, true>, 3, 1>;
extern template class MacroTrans<KernelTransFull<double, true>, 2, 4>;
extern template class MacroTrans<KernelTransFull<double, true>, 2, 3>;
extern template class MacroTrans<KernelTransFull<double, true>, 2, 2>;
extern template class MacroTrans<KernelTransFull<double, true>, 2, 1>;
extern template class MacroTrans<KernelTransFull<double, true>, 1, 4>;
extern template class MacroTrans<KernelTransFull<double, true>, 1, 3>;
extern template class MacroTrans<KernelTransFull<double, true>, 1, 2>;
extern template class MacroTrans<KernelTransFull<double, true>, 1, 1>;

extern template class MacroTrans<KernelTransFull<FloatComplex, true>, 4, 4>;
extern template class MacroTrans<KernelTransFull<FloatComplex, true>, 4, 3>;
extern template class MacroTrans<KernelTransFull<FloatComplex, true>, 4, 2>;
extern template class MacroTrans<KernelTransFull<FloatComplex, true>, 4, 1>;
extern template class MacroTrans<KernelTransFull<FloatComplex, true>, 3, 4>;
extern template class MacroTrans<KernelTransFull<FloatComplex, true>, 3, 3>;
extern template class MacroTrans<KernelTransFull<FloatComplex, true>, 3, 2>;
extern template class MacroTrans<KernelTransFull<FloatComplex, true>, 3, 1>;
extern template class MacroTrans<KernelTransFull<FloatComplex, true>, 2, 4>;
extern template class MacroTrans<KernelTransFull<FloatComplex, true>, 2, 3>;
extern template class MacroTrans<KernelTransFull<FloatComplex, true>, 2, 2>;
extern template class MacroTrans<KernelTransFull<FloatComplex, true>, 2, 1>;
extern template class MacroTrans<KernelTransFull<FloatComplex, true>, 1, 4>;
extern template class MacroTrans<KernelTransFull<FloatComplex, true>, 1, 3>;
extern template class MacroTrans<KernelTransFull<FloatComplex, true>, 1, 2>;
extern template class MacroTrans<KernelTransFull<FloatComplex, true>, 1, 1>;

extern template class MacroTrans<KernelTransFull<DoubleComplex, true>, 4, 4>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, true>, 4, 3>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, true>, 4, 2>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, true>, 4, 1>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, true>, 3, 4>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, true>, 3, 3>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, true>, 3, 2>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, true>, 3, 1>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, true>, 2, 4>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, true>, 2, 3>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, true>, 2, 2>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, true>, 2, 1>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, true>, 1, 4>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, true>, 1, 3>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, true>, 1, 2>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, true>, 1, 1>;

extern template class MacroTrans<KernelTransHalf<float, true>, 4, 1>;
extern template class MacroTrans<KernelTransHalf<float, true>, 3, 1>;
extern template class MacroTrans<KernelTransHalf<float, true>, 2, 1>;
extern template class MacroTrans<KernelTransHalf<float, true>, 1, 4>;
extern template class MacroTrans<KernelTransHalf<float, true>, 1, 3>;
extern template class MacroTrans<KernelTransHalf<float, true>, 1, 2>;
extern template class MacroTrans<KernelTransHalf<float, true>, 1, 1>;

extern template class MacroTrans<KernelTransHalf<double, true>, 4, 1>;
extern template class MacroTrans<KernelTransHalf<double, true>, 3, 1>;
extern template class MacroTrans<KernelTransHalf<double, true>, 2, 1>;
extern template class MacroTrans<KernelTransHalf<double, true>, 1, 4>;
extern template class MacroTrans<KernelTransHalf<double, true>, 1, 3>;
extern template class MacroTrans<KernelTransHalf<double, true>, 1, 2>;
extern template class MacroTrans<KernelTransHalf<double, true>, 1, 1>;

extern template class MacroTrans<KernelTransHalf<FloatComplex, true>, 4, 1>;
extern template class MacroTrans<KernelTransHalf<FloatComplex, true>, 3, 1>;
extern template class MacroTrans<KernelTransHalf<FloatComplex, true>, 2, 1>;
extern template class MacroTrans<KernelTransHalf<FloatComplex, true>, 1, 4>;
extern template class MacroTrans<KernelTransHalf<FloatComplex, true>, 1, 3>;
extern template class MacroTrans<KernelTransHalf<FloatComplex, true>, 1, 2>;
extern template class MacroTrans<KernelTransHalf<FloatComplex, true>, 1, 1>;

extern template class MacroTrans<KernelTransHalf<DoubleComplex, true>, 4, 1>;
extern template class MacroTrans<KernelTransHalf<DoubleComplex, true>, 3, 1>;
extern template class MacroTrans<KernelTransHalf<DoubleComplex, true>, 2, 1>;
extern template class MacroTrans<KernelTransHalf<DoubleComplex, true>, 1, 4>;
extern template class MacroTrans<KernelTransHalf<DoubleComplex, true>, 1, 3>;
extern template class MacroTrans<KernelTransHalf<DoubleComplex, true>, 1, 2>;
extern template class MacroTrans<KernelTransHalf<DoubleComplex, true>, 1, 1>;

extern template class MacroTrans<KernelTransFull<float, false>, 4, 4>;
extern template class MacroTrans<KernelTransFull<float, false>, 4, 3>;
extern template class MacroTrans<KernelTransFull<float, false>, 4, 2>;
extern template class MacroTrans<KernelTransFull<float, false>, 4, 1>;
extern template class MacroTrans<KernelTransFull<float, false>, 3, 4>;
extern template class MacroTrans<KernelTransFull<float, false>, 3, 3>;
extern template class MacroTrans<KernelTransFull<float, false>, 3, 2>;
extern template class MacroTrans<KernelTransFull<float, false>, 3, 1>;
extern template class MacroTrans<KernelTransFull<float, false>, 2, 4>;
extern template class MacroTrans<KernelTransFull<float, false>, 2, 3>;
extern template class MacroTrans<KernelTransFull<float, false>, 2, 2>;
extern template class MacroTrans<KernelTransFull<float, false>, 2, 1>;
extern template class MacroTrans<KernelTransFull<float, false>, 1, 4>;
extern template class MacroTrans<KernelTransFull<float, false>, 1, 3>;
extern template class MacroTrans<KernelTransFull<float, false>, 1, 2>;
extern template class MacroTrans<KernelTransFull<float, false>, 1, 1>;

extern template class MacroTrans<KernelTransFull<double, false>, 4, 4>;
extern template class MacroTrans<KernelTransFull<double, false>, 4, 3>;
extern template class MacroTrans<KernelTransFull<double, false>, 4, 2>;
extern template class MacroTrans<KernelTransFull<double, false>, 4, 1>;
extern template class MacroTrans<KernelTransFull<double, false>, 3, 4>;
extern template class MacroTrans<KernelTransFull<double, false>, 3, 3>;
extern template class MacroTrans<KernelTransFull<double, false>, 3, 2>;
extern template class MacroTrans<KernelTransFull<double, false>, 3, 1>;
extern template class MacroTrans<KernelTransFull<double, false>, 2, 4>;
extern template class MacroTrans<KernelTransFull<double, false>, 2, 3>;
extern template class MacroTrans<KernelTransFull<double, false>, 2, 2>;
extern template class MacroTrans<KernelTransFull<double, false>, 2, 1>;
extern template class MacroTrans<KernelTransFull<double, false>, 1, 4>;
extern template class MacroTrans<KernelTransFull<double, false>, 1, 3>;
extern template class MacroTrans<KernelTransFull<double, false>, 1, 2>;
extern template class MacroTrans<KernelTransFull<double, false>, 1, 1>;

extern template class MacroTrans<KernelTransFull<FloatComplex, false>, 4, 4>;
extern template class MacroTrans<KernelTransFull<FloatComplex, false>, 4, 3>;
extern template class MacroTrans<KernelTransFull<FloatComplex, false>, 4, 2>;
extern template class MacroTrans<KernelTransFull<FloatComplex, false>, 4, 1>;
extern template class MacroTrans<KernelTransFull<FloatComplex, false>, 3, 4>;
extern template class MacroTrans<KernelTransFull<FloatComplex, false>, 3, 3>;
extern template class MacroTrans<KernelTransFull<FloatComplex, false>, 3, 2>;
extern template class MacroTrans<KernelTransFull<FloatComplex, false>, 3, 1>;
extern template class MacroTrans<KernelTransFull<FloatComplex, false>, 2, 4>;
extern template class MacroTrans<KernelTransFull<FloatComplex, false>, 2, 3>;
extern template class MacroTrans<KernelTransFull<FloatComplex, false>, 2, 2>;
extern template class MacroTrans<KernelTransFull<FloatComplex, false>, 2, 1>;
extern template class MacroTrans<KernelTransFull<FloatComplex, false>, 1, 4>;
extern template class MacroTrans<KernelTransFull<FloatComplex, false>, 1, 3>;
extern template class MacroTrans<KernelTransFull<FloatComplex, false>, 1, 2>;
extern template class MacroTrans<KernelTransFull<FloatComplex, false>, 1, 1>;

extern template class MacroTrans<KernelTransFull<DoubleComplex, false>, 4, 4>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, false>, 4, 3>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, false>, 4, 2>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, false>, 4, 1>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, false>, 3, 4>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, false>, 3, 3>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, false>, 3, 2>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, false>, 3, 1>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, false>, 2, 4>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, false>, 2, 3>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, false>, 2, 2>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, false>, 2, 1>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, false>, 1, 4>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, false>, 1, 3>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, false>, 1, 2>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, false>, 1, 1>;

extern template class MacroTrans<KernelTransHalf<float, false>, 4, 1>;
extern template class MacroTrans<KernelTransHalf<float, false>, 3, 1>;
extern template class MacroTrans<KernelTransHalf<float, false>, 2, 1>;
extern template class MacroTrans<KernelTransHalf<float, false>, 1, 4>;
extern template class MacroTrans<KernelTransHalf<float, false>, 1, 3>;
extern template class MacroTrans<KernelTransHalf<float, false>, 1, 2>;
extern template class MacroTrans<KernelTransHalf<float, false>, 1, 1>;

extern template class MacroTrans<KernelTransHalf<double, false>, 4, 1>;
extern template class MacroTrans<KernelTransHalf<double, false>, 3, 1>;
extern template class MacroTrans<KernelTransHalf<double, false>, 2, 1>;
extern template class MacroTrans<KernelTransHalf<double, false>, 1, 4>;
extern template class MacroTrans<KernelTransHalf<double, false>, 1, 3>;
extern template class MacroTrans<KernelTransHalf<double, false>, 1, 2>;
extern template class MacroTrans<KernelTransHalf<double, false>, 1, 1>;

extern template class MacroTrans<KernelTransHalf<FloatComplex, false>, 4, 1>;
extern template class MacroTrans<KernelTransHalf<FloatComplex, false>, 3, 1>;
extern template class MacroTrans<KernelTransHalf<FloatComplex, false>, 2, 1>;
extern template class MacroTrans<KernelTransHalf<FloatComplex, false>, 1, 4>;
extern template class MacroTrans<KernelTransHalf<FloatComplex, false>, 1, 3>;
extern template class MacroTrans<KernelTransHalf<FloatComplex, false>, 1, 2>;
extern template class MacroTrans<KernelTransHalf<FloatComplex, false>, 1, 1>;

extern template class MacroTrans<KernelTransHalf<DoubleComplex, false>, 4, 1>;
extern template class MacroTrans<KernelTransHalf<DoubleComplex, false>, 3, 1>;
extern template class MacroTrans<KernelTransHalf<DoubleComplex, false>, 2, 1>;
extern template class MacroTrans<KernelTransHalf<DoubleComplex, false>, 1, 4>;
extern template class MacroTrans<KernelTransHalf<DoubleComplex, false>, 1, 3>;
extern template class MacroTrans<KernelTransHalf<DoubleComplex, false>, 1, 2>;
extern template class MacroTrans<KernelTransHalf<DoubleComplex, false>, 1, 1>;


/*
 * Explicit template instantiation for class MacroTransLinear
 */
extern template class MacroTransLinear<float, true>;
extern template class MacroTransLinear<double, true>;
extern template class MacroTransLinear<FloatComplex, true>;
extern template class MacroTransLinear<DoubleComplex, true>;

extern template class MacroTransLinear<float, false>;
extern template class MacroTransLinear<double, false>;
extern template class MacroTransLinear<FloatComplex, false>;
extern template class MacroTransLinear<DoubleComplex, false>;


/*
 * Explicit template instantiation for class MacroTransScalar
 */
extern template class MacroTransScalar<float, true>;
extern template class MacroTransScalar<double, true>;
extern template class MacroTransScalar<FloatComplex, true>;
extern template class MacroTransScalar<DoubleComplex, true>;

extern template class MacroTransScalar<float, false>;
extern template class MacroTransScalar<double, false>;
extern template class MacroTransScalar<FloatComplex, false>;
extern template class MacroTransScalar<DoubleComplex, false>;

#endif // HPTT_KERNELS_MACRO_KERNEL_TRANS_TCC_
