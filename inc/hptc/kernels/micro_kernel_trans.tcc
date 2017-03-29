#pragma once
#ifndef HPTC_KERNELS_MICRO_KERNEL_TRANS_TCC_
#define HPTC_KERNELS_MICRO_KERNEL_TRANS_TCC_

/*
 * Explicit template instantiation declaration
 */
extern template class KernelTransProxy<float, KernelTypeTrans::KERNEL_FULL>;
extern template class KernelTransProxy<float, KernelTypeTrans::KERNEL_HALF>;
extern template class KernelTransProxy<float, KernelTypeTrans::KERNEL_LINE>;
extern template class KernelTransProxy<double, KernelTypeTrans::KERNEL_FULL>;
extern template class KernelTransProxy<double, KernelTypeTrans::KERNEL_HALF>;
extern template class KernelTransProxy<double, KernelTypeTrans::KERNEL_LINE>;
extern template class KernelTransProxy<FloatComplex,
    KernelTypeTrans::KERNEL_FULL>;
extern template class KernelTransProxy<FloatComplex,
    KernelTypeTrans::KERNEL_HALF>;
extern template class KernelTransProxy<FloatComplex,
    KernelTypeTrans::KERNEL_LINE>;
extern template class KernelTransProxy<DoubleComplex,
    KernelTypeTrans::KERNEL_FULL>;
extern template class KernelTransProxy<DoubleComplex,
    KernelTypeTrans::KERNEL_HALF>;
extern template class KernelTransProxy<DoubleComplex,
    KernelTypeTrans::KERNEL_LINE>;

#endif // HPTC_KERNELS_MICRO_KERNEL_TRANS_TCC_
