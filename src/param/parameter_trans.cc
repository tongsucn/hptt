#include <hptc/param/parameter_trans.h>


namespace hptc {

/*
 * Explicit template instantiation for stract ParamTrans
 */
template struct ParamTrans<
    TensorWrapper<float, 2, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<float, 2, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<float, 2, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<float, 2, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BOTH>;
template struct ParamTrans<
    TensorWrapper<float, 2, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<float, 2, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<float, 2, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<float, 2, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BOTH>;

template struct ParamTrans<
    TensorWrapper<float, 3, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<float, 3, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<float, 3, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<float, 3, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BOTH>;
template struct ParamTrans<
    TensorWrapper<float, 3, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<float, 3, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<float, 3, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<float, 3, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BOTH>;

template struct ParamTrans<
    TensorWrapper<float, 4, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<float, 4, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<float, 4, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<float, 4, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BOTH>;
template struct ParamTrans<
    TensorWrapper<float, 4, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<float, 4, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<float, 4, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<float, 4, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BOTH>;

template struct ParamTrans<
    TensorWrapper<float, 5, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<float, 5, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<float, 5, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<float, 5, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BOTH>;
template struct ParamTrans<
    TensorWrapper<float, 5, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<float, 5, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<float, 5, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<float, 5, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BOTH>;

template struct ParamTrans<
    TensorWrapper<float, 6, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<float, 6, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<float, 6, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<float, 6, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BOTH>;
template struct ParamTrans<
    TensorWrapper<float, 6, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<float, 6, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<float, 6, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<float, 6, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BOTH>;

template struct ParamTrans<
    TensorWrapper<float, 7, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<float, 7, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<float, 7, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<float, 7, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BOTH>;
template struct ParamTrans<
    TensorWrapper<float, 7, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<float, 7, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<float, 7, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<float, 7, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BOTH>;

template struct ParamTrans<
    TensorWrapper<float, 8, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<float, 8, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<float, 8, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<float, 8, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BOTH>;
template struct ParamTrans<
    TensorWrapper<float, 8, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<float, 8, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<float, 8, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<float, 8, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BOTH>;

template struct ParamTrans<
    TensorWrapper<double, 2, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<double, 2, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<double, 2, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<double, 2, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BOTH>;
template struct ParamTrans<
    TensorWrapper<double, 2, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<double, 2, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<double, 2, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<double, 2, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BOTH>;

template struct ParamTrans<
    TensorWrapper<double, 3, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<double, 3, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<double, 3, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<double, 3, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BOTH>;
template struct ParamTrans<
    TensorWrapper<double, 3, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<double, 3, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<double, 3, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<double, 3, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BOTH>;

template struct ParamTrans<
    TensorWrapper<double, 4, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<double, 4, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<double, 4, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<double, 4, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BOTH>;
template struct ParamTrans<
    TensorWrapper<double, 4, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<double, 4, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<double, 4, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<double, 4, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BOTH>;

template struct ParamTrans<
    TensorWrapper<double, 5, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<double, 5, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<double, 5, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<double, 5, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BOTH>;
template struct ParamTrans<
    TensorWrapper<double, 5, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<double, 5, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<double, 5, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<double, 5, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BOTH>;

template struct ParamTrans<
    TensorWrapper<double, 6, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<double, 6, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<double, 6, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<double, 6, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BOTH>;
template struct ParamTrans<
    TensorWrapper<double, 6, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<double, 6, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<double, 6, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<double, 6, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BOTH>;

template struct ParamTrans<
    TensorWrapper<double, 7, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<double, 7, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<double, 7, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<double, 7, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BOTH>;
template struct ParamTrans<
    TensorWrapper<double, 7, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<double, 7, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<double, 7, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<double, 7, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BOTH>;

template struct ParamTrans<
    TensorWrapper<double, 8, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<double, 8, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<double, 8, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<double, 8, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BOTH>;
template struct ParamTrans<
    TensorWrapper<double, 8, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<double, 8, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<double, 8, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<double, 8, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BOTH>;

template struct ParamTrans<
    TensorWrapper<FloatComplex, 2, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 2, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 2, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 2, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BOTH>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 2, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 2, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 2, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 2, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BOTH>;

template struct ParamTrans<
    TensorWrapper<FloatComplex, 3, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 3, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 3, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 3, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BOTH>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 3, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 3, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 3, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 3, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BOTH>;

template struct ParamTrans<
    TensorWrapper<FloatComplex, 4, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 4, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 4, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 4, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BOTH>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 4, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 4, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 4, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 4, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BOTH>;

template struct ParamTrans<
    TensorWrapper<FloatComplex, 5, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 5, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 5, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 5, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BOTH>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 5, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 5, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 5, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 5, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BOTH>;

template struct ParamTrans<
    TensorWrapper<FloatComplex, 6, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 6, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 6, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 6, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BOTH>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 6, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 6, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 6, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 6, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BOTH>;

template struct ParamTrans<
    TensorWrapper<FloatComplex, 7, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 7, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 7, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 7, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BOTH>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 7, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 7, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 7, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 7, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BOTH>;

template struct ParamTrans<
    TensorWrapper<FloatComplex, 8, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 8, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 8, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 8, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BOTH>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 8, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 8, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 8, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<FloatComplex, 8, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BOTH>;

template struct ParamTrans<
    TensorWrapper<DoubleComplex, 2, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 2, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 2, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 2, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BOTH>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 2, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 2, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 2, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 2, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BOTH>;

template struct ParamTrans<
    TensorWrapper<DoubleComplex, 3, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 3, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 3, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 3, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BOTH>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 3, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 3, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 3, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 3, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BOTH>;

template struct ParamTrans<
    TensorWrapper<DoubleComplex, 4, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 4, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 4, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 4, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BOTH>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 4, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 4, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 4, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 4, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BOTH>;

template struct ParamTrans<
    TensorWrapper<DoubleComplex, 5, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 5, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 5, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 5, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BOTH>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 5, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 5, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 5, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 5, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BOTH>;

template struct ParamTrans<
    TensorWrapper<DoubleComplex, 6, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 6, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 6, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 6, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BOTH>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 6, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 6, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 6, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 6, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BOTH>;

template struct ParamTrans<
    TensorWrapper<DoubleComplex, 7, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 7, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 7, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 7, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BOTH>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 7, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 7, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 7, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 7, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BOTH>;

template struct ParamTrans<
    TensorWrapper<DoubleComplex, 8, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 8, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 8, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 8, MemLayout::COL_MAJOR>,
    CoefUsageTrans::USE_BOTH>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 8, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_NONE>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 8, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_ALPHA>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 8, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BETA>;
template struct ParamTrans<
    TensorWrapper<DoubleComplex, 8, MemLayout::ROW_MAJOR>,
    CoefUsageTrans::USE_BOTH>;

}
