#include <hptc/kernels/kernel_trans.h>

#include <hptc/types.h>


namespace hptc {

/*
 * Explicit template instantiation definition for struct KernelPackTrans
 */
template struct KernelPackTrans<float>;
template struct KernelPackTrans<double>;
template struct KernelPackTrans<FloatComplex>;
template struct KernelPackTrans<DoubleComplex>;

}
