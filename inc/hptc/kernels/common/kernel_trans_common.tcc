#pragma once
#ifndef HPTC_KERNELS_COMMON_KERNEL_TRANS_COMMON_TCC_
#define HPTC_KERNELS_COMMON_KERNEL_TRANS_COMMON_TCC_

/*
 * Explicit instantiation declaration for struct KernelTransCommon
 */
extern template struct KernelTransCommon<float, CoefUsageTrans::USE_NONE,
    KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransCommon<float, CoefUsageTrans::USE_NONE,
    KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransCommon<float, CoefUsageTrans::USE_ALPHA,
    KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransCommon<float, CoefUsageTrans::USE_ALPHA,
    KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransCommon<float, CoefUsageTrans::USE_BETA,
    KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransCommon<float, CoefUsageTrans::USE_BETA,
    KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransCommon<float, CoefUsageTrans::USE_BOTH,
    KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransCommon<float, CoefUsageTrans::USE_BOTH,
    KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransCommon<double, CoefUsageTrans::USE_NONE,
    KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransCommon<double, CoefUsageTrans::USE_NONE,
    KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransCommon<double, CoefUsageTrans::USE_ALPHA,
    KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransCommon<double, CoefUsageTrans::USE_ALPHA,
    KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransCommon<double, CoefUsageTrans::USE_BETA,
    KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransCommon<double, CoefUsageTrans::USE_BETA,
    KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransCommon<double, CoefUsageTrans::USE_BOTH,
    KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransCommon<double, CoefUsageTrans::USE_BOTH,
    KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransCommon<FloatComplex,
    CoefUsageTrans::USE_NONE, KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransCommon<FloatComplex,
    CoefUsageTrans::USE_NONE, KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransCommon<FloatComplex,
    CoefUsageTrans::USE_ALPHA, KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransCommon<FloatComplex,
    CoefUsageTrans::USE_ALPHA, KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransCommon<FloatComplex,
    CoefUsageTrans::USE_BETA, KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransCommon<FloatComplex,
    CoefUsageTrans::USE_BETA, KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransCommon<FloatComplex,
    CoefUsageTrans::USE_BOTH, KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransCommon<FloatComplex,
    CoefUsageTrans::USE_BOTH, KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransCommon<DoubleComplex,
    CoefUsageTrans::USE_NONE, KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransCommon<DoubleComplex,
    CoefUsageTrans::USE_NONE, KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransCommon<DoubleComplex,
    CoefUsageTrans::USE_ALPHA, KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransCommon<DoubleComplex,
    CoefUsageTrans::USE_ALPHA, KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransCommon<DoubleComplex,
    CoefUsageTrans::USE_BETA, KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransCommon<DoubleComplex,
    CoefUsageTrans::USE_BETA, KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransCommon<DoubleComplex,
    CoefUsageTrans::USE_BOTH, KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransCommon<DoubleComplex,
    CoefUsageTrans::USE_BOTH, KernelTypeTrans::KERNEL_HALF>;

#endif // HPTC_KERNELS_COMMON_KERNEL_TRANS_COMMON_TCC_
