#include <hptc/param/parameter_trans.h>

#include <hptc/types.h>
#include <hptc/config/config_trans.h>


namespace hptc {

/*
 * Explicit template instantiation for stract ParamTrans
 */
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 2, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 3, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 4, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 5, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 6, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 7, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 8, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;

template struct ParamTrans<
    TensorWrapper<DoubleComplex, 2, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 3, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 4, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 5, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 6, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 7, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 8, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;

}
