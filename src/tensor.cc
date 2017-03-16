#include <hptc/tensor.h>

namespace hptc {

/*
 * Explicit template instantiation for class TensorWrapper
 */
template class TensorWrapper<float, 2, MemLayout::COL_MAJOR>;
template class TensorWrapper<float, 2, MemLayout::ROW_MAJOR>;
template class TensorWrapper<float, 3, MemLayout::COL_MAJOR>;
template class TensorWrapper<float, 3, MemLayout::ROW_MAJOR>;
template class TensorWrapper<float, 4, MemLayout::COL_MAJOR>;
template class TensorWrapper<float, 4, MemLayout::ROW_MAJOR>;
template class TensorWrapper<float, 5, MemLayout::COL_MAJOR>;
template class TensorWrapper<float, 5, MemLayout::ROW_MAJOR>;
template class TensorWrapper<float, 6, MemLayout::COL_MAJOR>;
template class TensorWrapper<float, 6, MemLayout::ROW_MAJOR>;
template class TensorWrapper<float, 7, MemLayout::COL_MAJOR>;
template class TensorWrapper<float, 7, MemLayout::ROW_MAJOR>;
template class TensorWrapper<float, 8, MemLayout::COL_MAJOR>;
template class TensorWrapper<float, 8, MemLayout::ROW_MAJOR>;

template class TensorWrapper<double, 2, MemLayout::COL_MAJOR>;
template class TensorWrapper<double, 2, MemLayout::ROW_MAJOR>;
template class TensorWrapper<double, 3, MemLayout::COL_MAJOR>;
template class TensorWrapper<double, 3, MemLayout::ROW_MAJOR>;
template class TensorWrapper<double, 4, MemLayout::COL_MAJOR>;
template class TensorWrapper<double, 4, MemLayout::ROW_MAJOR>;
template class TensorWrapper<double, 5, MemLayout::COL_MAJOR>;
template class TensorWrapper<double, 5, MemLayout::ROW_MAJOR>;
template class TensorWrapper<double, 6, MemLayout::COL_MAJOR>;
template class TensorWrapper<double, 6, MemLayout::ROW_MAJOR>;
template class TensorWrapper<double, 7, MemLayout::COL_MAJOR>;
template class TensorWrapper<double, 7, MemLayout::ROW_MAJOR>;
template class TensorWrapper<double, 8, MemLayout::COL_MAJOR>;
template class TensorWrapper<double, 8, MemLayout::ROW_MAJOR>;

template class TensorWrapper<FloatComplex, 2, MemLayout::COL_MAJOR>;
template class TensorWrapper<FloatComplex, 2, MemLayout::ROW_MAJOR>;
template class TensorWrapper<FloatComplex, 3, MemLayout::COL_MAJOR>;
template class TensorWrapper<FloatComplex, 3, MemLayout::ROW_MAJOR>;
template class TensorWrapper<FloatComplex, 4, MemLayout::COL_MAJOR>;
template class TensorWrapper<FloatComplex, 4, MemLayout::ROW_MAJOR>;
template class TensorWrapper<FloatComplex, 5, MemLayout::COL_MAJOR>;
template class TensorWrapper<FloatComplex, 5, MemLayout::ROW_MAJOR>;
template class TensorWrapper<FloatComplex, 6, MemLayout::COL_MAJOR>;
template class TensorWrapper<FloatComplex, 6, MemLayout::ROW_MAJOR>;
template class TensorWrapper<FloatComplex, 7, MemLayout::COL_MAJOR>;
template class TensorWrapper<FloatComplex, 7, MemLayout::ROW_MAJOR>;
template class TensorWrapper<FloatComplex, 8, MemLayout::COL_MAJOR>;
template class TensorWrapper<FloatComplex, 8, MemLayout::ROW_MAJOR>;

template class TensorWrapper<DoubleComplex, 2, MemLayout::COL_MAJOR>;
template class TensorWrapper<DoubleComplex, 2, MemLayout::ROW_MAJOR>;
template class TensorWrapper<DoubleComplex, 3, MemLayout::COL_MAJOR>;
template class TensorWrapper<DoubleComplex, 3, MemLayout::ROW_MAJOR>;
template class TensorWrapper<DoubleComplex, 4, MemLayout::COL_MAJOR>;
template class TensorWrapper<DoubleComplex, 4, MemLayout::ROW_MAJOR>;
template class TensorWrapper<DoubleComplex, 5, MemLayout::COL_MAJOR>;
template class TensorWrapper<DoubleComplex, 5, MemLayout::ROW_MAJOR>;
template class TensorWrapper<DoubleComplex, 6, MemLayout::COL_MAJOR>;
template class TensorWrapper<DoubleComplex, 6, MemLayout::ROW_MAJOR>;
template class TensorWrapper<DoubleComplex, 7, MemLayout::COL_MAJOR>;
template class TensorWrapper<DoubleComplex, 7, MemLayout::ROW_MAJOR>;
template class TensorWrapper<DoubleComplex, 8, MemLayout::COL_MAJOR>;
template class TensorWrapper<DoubleComplex, 8, MemLayout::ROW_MAJOR>;

}
