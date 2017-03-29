#include <hptc/kernels/kernel_trans.h>

#include <hptc/types.h>
#include <hptc/util/util_trans.h>


namespace hptc {

/*
 * Explicit instantiation for struct KernelPackTrans
 */
template struct KernelPackTrans<float>;
template struct KernelPackTrans<double>;
template struct KernelPackTrans<FloatComplex>;
template struct KernelPackTrans<DoubleComplex>;

}
