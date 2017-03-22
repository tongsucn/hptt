#include <hptc/kernels/macro_kernel_trans.h>

#include <hptc/types.h>
#include <hptc/util/util_trans.h>


namespace hptc {

/*
 * Explicit template instantiation for class MacroTransVec
 */
template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_NONE>, 4, 4>;
template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_NONE>, 4, 3>;
template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_NONE>, 4, 2>;
template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_NONE>, 4, 1>;
template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_NONE>, 3, 4>;
template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_NONE>, 3, 3>;
template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_NONE>, 3, 2>;
template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_NONE>, 3, 1>;
template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_NONE>, 2, 4>;
template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_NONE>, 2, 3>;
template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_NONE>, 2, 2>;
template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_NONE>, 2, 1>;
template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_NONE>, 1, 4>;
template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_NONE>, 1, 3>;
template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_NONE>, 1, 2>;
template class MacroTransVec<
    KernelTransFull<float, CoefUsageTrans::USE_NONE>, 1, 1>;

template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_NONE>, 4, 4>;
template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_NONE>, 4, 3>;
template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_NONE>, 4, 2>;
template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_NONE>, 4, 1>;
template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_NONE>, 3, 4>;
template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_NONE>, 3, 3>;
template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_NONE>, 3, 2>;
template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_NONE>, 3, 1>;
template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_NONE>, 2, 4>;
template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_NONE>, 2, 3>;
template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_NONE>, 2, 2>;
template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_NONE>, 2, 1>;
template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_NONE>, 1, 4>;
template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_NONE>, 1, 3>;
template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_NONE>, 1, 2>;
template class MacroTransVec<
    KernelTransFull<double, CoefUsageTrans::USE_NONE>, 1, 1>;

template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_NONE>, 4, 4>;
template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_NONE>, 4, 3>;
template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_NONE>, 4, 2>;
template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_NONE>, 4, 1>;
template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_NONE>, 3, 4>;
template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_NONE>, 3, 3>;
template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_NONE>, 3, 2>;
template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_NONE>, 3, 1>;
template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_NONE>, 2, 4>;
template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_NONE>, 2, 3>;
template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_NONE>, 2, 2>;
template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_NONE>, 2, 1>;
template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_NONE>, 1, 4>;
template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_NONE>, 1, 3>;
template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_NONE>, 1, 2>;
template class MacroTransVec<
    KernelTransFull<FloatComplex, CoefUsageTrans::USE_NONE>, 1, 1>;

template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_NONE>, 4, 4>;
template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_NONE>, 4, 3>;
template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_NONE>, 4, 2>;
template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_NONE>, 4, 1>;
template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_NONE>, 3, 4>;
template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_NONE>, 3, 3>;
template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_NONE>, 3, 2>;
template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_NONE>, 3, 1>;
template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_NONE>, 2, 4>;
template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_NONE>, 2, 3>;
template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_NONE>, 2, 2>;
template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_NONE>, 2, 1>;
template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_NONE>, 1, 4>;
template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_NONE>, 1, 3>;
template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_NONE>, 1, 2>;
template class MacroTransVec<
    KernelTransFull<DoubleComplex, CoefUsageTrans::USE_NONE>, 1, 1>;


template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_NONE>, 4, 1>;
template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_NONE>, 3, 1>;
template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_NONE>, 2, 1>;
template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_NONE>, 1, 4>;
template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_NONE>, 1, 3>;
template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_NONE>, 1, 2>;
template class MacroTransVec<
    KernelTransHalf<float, CoefUsageTrans::USE_NONE>, 1, 1>;

template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_NONE>, 4, 1>;
template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_NONE>, 3, 1>;
template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_NONE>, 2, 1>;
template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_NONE>, 1, 4>;
template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_NONE>, 1, 3>;
template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_NONE>, 1, 2>;
template class MacroTransVec<
    KernelTransHalf<double, CoefUsageTrans::USE_NONE>, 1, 1>;

template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_NONE>, 4, 1>;
template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_NONE>, 3, 1>;
template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_NONE>, 2, 1>;
template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_NONE>, 1, 4>;
template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_NONE>, 1, 3>;
template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_NONE>, 1, 2>;
template class MacroTransVec<
    KernelTransHalf<FloatComplex, CoefUsageTrans::USE_NONE>, 1, 1>;

template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_NONE>, 4, 1>;
template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_NONE>, 3, 1>;
template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_NONE>, 2, 1>;
template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_NONE>, 1, 4>;
template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_NONE>, 1, 3>;
template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_NONE>, 1, 2>;
template class MacroTransVec<
    KernelTransHalf<DoubleComplex, CoefUsageTrans::USE_NONE>, 1, 1>;

}
