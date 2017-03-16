#include <hptc/kernels/kernel_trans.h>

#include <hptc/types.h>
#include <hptc/config/config_trans.h>


namespace hptc {

/*
 * Explicit instantiation for struct KernelPackTrans
 */
template struct KernelPackTrans<float, CoefUsageTrans::USE_NONE>;
template struct KernelPackTrans<float, CoefUsageTrans::USE_ALPHA>;
template struct KernelPackTrans<float, CoefUsageTrans::USE_BETA>;
template struct KernelPackTrans<float, CoefUsageTrans::USE_BOTH>;

template struct KernelPackTrans<double, CoefUsageTrans::USE_NONE>;
template struct KernelPackTrans<double, CoefUsageTrans::USE_ALPHA>;
template struct KernelPackTrans<double, CoefUsageTrans::USE_BETA>;
template struct KernelPackTrans<double, CoefUsageTrans::USE_BOTH>;

template struct KernelPackTrans<FloatComplex, CoefUsageTrans::USE_NONE>;
template struct KernelPackTrans<FloatComplex, CoefUsageTrans::USE_ALPHA>;
template struct KernelPackTrans<FloatComplex, CoefUsageTrans::USE_BETA>;
template struct KernelPackTrans<FloatComplex, CoefUsageTrans::USE_BOTH>;

template struct KernelPackTrans<DoubleComplex, CoefUsageTrans::USE_NONE>;
template struct KernelPackTrans<DoubleComplex, CoefUsageTrans::USE_ALPHA>;
template struct KernelPackTrans<DoubleComplex, CoefUsageTrans::USE_BETA>;
template struct KernelPackTrans<DoubleComplex, CoefUsageTrans::USE_BOTH>;

}
