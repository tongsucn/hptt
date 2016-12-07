#pragma once
#ifndef HPTC_OPERATIONS_OPERATION_TRANS_H_
#define HPTC_OPERATIONS_OPERATION_TRANS_H_

#include <hptc/types.h>
#include <hptc/param/parameters.h>
#include <hptc/kernels/kernel_trans.h>
#include <hptc/operations/operation_base.h>


namespace hptc {

template <typename FloatType,
          GenNumType HEIGHT,
          GenNumType WIDTH = HEIGHT>
class OpMacroTrans : public OpMacro<FloatType, ParamTrans> {
public:
  OpMacroTrans(const std::shared_ptr<ParamTrans<FloatType>> &param);

  OpMacroTrans(const OpMacroTrans &operation) = delete;
  OpMacroTrans &operator=(const OpMacroTrans &operation) = delete;

  virtual ~OpMacroTrans();

  virtual INLINE void exec() final;

private:
  KernelTransBase<FloatType> *kernel_;
  GenNumType kernel_size_;
};


template <GenNumType ROWS,
          GenNumType COLS>
struct DualCounter {
};


template <typename FloatType,
          GenNumType ROWS,
          GenNumType COLS>
INLINE void kernel_tiler(DualCounter<ROWS, COLS>, GenNumType kernel_size,
    KernelTransBase<FloatType> *kernel,
    const FloatType *input_data, FloatType *output_data,
    TensorIdx input_offset, TensorIdx output_offset,
    DeducedFloatType<FloatType> alpha, DeducedFloatType<FloatType> beta);


template <typename FloatType,
          GenNumType COLS>
INLINE void kernel_tiler(DualCounter<0, COLS>, GenNumType kernel_size,
    KernelTransBase<FloatType> *kernel,
    const FloatType *input_data, FloatType *output_data,
    TensorIdx input_offset, TensorIdx output_offset,
    DeducedFloatType<FloatType> alpha, DeducedFloatType<FloatType> beta);


template <typename FloatType,
          GenNumType ROWS,
          GenNumType COLS>
INLINE void cols_tiler(DualCounter<ROWS, COLS>, GenNumType kernel_size,
    KernelTransBase<FloatType> *kernel,
    const FloatType *input_data, FloatType *output_data,
    TensorIdx input_offset, TensorIdx output_offset,
    DeducedFloatType<FloatType> alpha, DeducedFloatType<FloatType> beta);


template <typename FloatType,
          GenNumType ROWS>
INLINE void cols_tiler(DualCounter<ROWS, 0>, GenNumType kernel_size,
    KernelTransBase<FloatType> *kernel,
    const FloatType *input_data, FloatType *output_data,
    TensorIdx input_offset, TensorIdx output_offset,
    DeducedFloatType<FloatType> alpha, DeducedFloatType<FloatType> beta);

/*
 * Import implementation
 */
#include "operation_trans.tcc"


/*
 * Transpose Macro kernels template instantiation
 */
template <typename FloatType>
using OpMacroTrans1x1 = OpMacroTrans<FloatType, 1, 1>;


template <typename FloatType>
using OpMacroTrans1x2 = OpMacroTrans<FloatType, 1, 2>;


template <typename FloatType>
using OpMacroTrans1x3 = OpMacroTrans<FloatType, 1, 3>;


template <typename FloatType>
using OpMacroTrans1x4 = OpMacroTrans<FloatType, 1, 4>;


template <typename FloatType>
using OpMacroTrans2x1 = OpMacroTrans<FloatType, 2, 1>;


template <typename FloatType>
using OpMacroTrans2x2 = OpMacroTrans<FloatType, 2, 2>;


template <typename FloatType>
using OpMacroTrans2x3 = OpMacroTrans<FloatType, 2, 3>;


template <typename FloatType>
using OpMacroTrans2x4 = OpMacroTrans<FloatType, 2, 4>;


template <typename FloatType>
using OpMacroTrans3x1 = OpMacroTrans<FloatType, 3, 1>;


template <typename FloatType>
using OpMacroTrans3x2 = OpMacroTrans<FloatType, 3, 2>;


template <typename FloatType>
using OpMacroTrans3x3 = OpMacroTrans<FloatType, 3, 3>;


template <typename FloatType>
using OpMacroTrans3x4 = OpMacroTrans<FloatType, 3, 4>;


template <typename FloatType>
using OpMacroTrans4x1 = OpMacroTrans<FloatType, 4, 1>;


template <typename FloatType>
using OpMacroTrans4x2 = OpMacroTrans<FloatType, 4, 2>;


template <typename FloatType>
using OpMacroTrans4x3 = OpMacroTrans<FloatType, 4, 3>;


template <typename FloatType>
using OpMacroTrans4x4 = OpMacroTrans<FloatType, 4, 4>;

}

#endif // HPTC_OPERATIONS_OPERATION_TRANS_H_
