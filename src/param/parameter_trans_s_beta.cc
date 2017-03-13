#include <hptc/param/parameter_trans.h>

#include <hptc/types.h>
#include <hptc/config/config_trans.h>


namespace hptc {

/*
 * Explicit template instantiation for stract ParamTrans
 */
template struct ParamTrans<
    TensorWrapper<float, 2, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<float, 3, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<float, 4, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<float, 5, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<float, 6, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<float, 7, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<float, 8, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BETA>;

template struct ParamTrans<
    TensorWrapper<float, 2, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<float, 3, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<float, 4, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<float, 5, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<float, 6, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<float, 7, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<float, 8, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BETA>;

}
