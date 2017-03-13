#include <hptc/param/parameter_trans.h>

#include <hptc/types.h>
#include <hptc/config/config_trans.h>


namespace hptc {

/*
 * Explicit template instantiation for stract ParamTrans
 */
template struct ParamTrans<
    TensorWrapper<FloatComplex, 2, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 3, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 4, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 5, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 6, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 7, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 8, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_NONE>;

template struct ParamTrans<
    TensorWrapper<FloatComplex, 2, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 3, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 4, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 5, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 6, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 7, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 8, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_NONE>;

}
