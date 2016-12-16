#pragma once
#ifndef HPTC_KERNELS_MACRO_KERNEL_TRANS_H_
#define HPTC_KERNELS_MACRO_KERNEL_TRANS_H_

#include <memory>

#include <hptc/kernels/kernel_trans.h>


namespace hptc {

template <typename FloatType,
          GenNumType HEIGHT,
          GenNumType WIDTH,
          KernelTransType KERNEL_TYPE = KernelTransType::KERNEL_FULL,
          MemLayout LAYOUT = MemLayout::COL_MAJOR>
class MacroTrans {
public:
  MacroTrans(const std::shared_ptr<ParamTrans<FloatType, LAYOUT>> &param);

  MacroTrans(const MacroTrans &operation) = delete;
  MacroTrans(const MacroTrans &operation) = delete;

  virtual ~MacroTransFull();

  virtual INLINE void exec() final;

private:
  std::shared_ptr<ParamTrans<FloatType, LAYOUT>> param;
  KernelTransBase<FloatType, KERNEL_TYPE> *kernel_;
  GenNumType kernel_size_;
};


template <GenNumType ROWS,
          GenNumType COLS>
struct DualCounter {
};


template <typename FloatType,
          GenNumType ROWS,
          GenNumType COLS,
          MemLayout LAYOUT,
          KernelTransType KERNEL_TYPE>
INLINE void row_tiler(DualCounter<ROWS, COLS>, GenNumType kernel_size,
    KernelTransBase<FloatType, KERNEL_TYPE> *kernel,
    const FloatType *input_data, FloatType *output_data,
    TensorIdx input_stride, TensorIdx output_stride);


template <typename FloatType,
          GenNumType COLS,
          MemLayout LAYOUT,
          KernelTransType KERNEL_TYPE>
INLINE void row_tiler(DualCounter<0, COLS>, GenNumType kernel_size,
    KernelTransBase<FloatType, KERNEL_TYPE> *kernel,
    const FloatType *input_data, FloatType *output_data,
    TensorIdx input_stride, TensorIdx output_stride);


template <typename FloatType,
          GenNumType ROWS,
          GenNumType COLS,
          MemLayout LAYOUT,
          KernelTransType KERNEL_TYPE>
INLINE void col_tiler(DualCounter<ROWS, COLS>, GenNumType kernel_size,
    KernelTransBase<FloatType, KERNEL_TYPE> *kernel,
    const FloatType *input_data, FloatType *output_data,
    TensorIdx input_stride, TensorIdx output_stride);


template <typename FloatType,
          GenNumType ROWS,
          MemLayout LAYOUT,
          KernelTransType KERNEL_TYPE>
INLINE void col_tiler(DualCounter<ROWS, 0>, GenNumType kernel_size,
    KernelTransBase<FloatType, KERNEL_TYPE> *kernel,
    const FloatType *input_data, FloatType *output_data,
    TensorIdx input_stride, TensorIdx output_stride);


/*
 * Import implementation
 */
#include "macro_kernel_trans.tcc"

}

#endif // HPTC_KERNELS_MACRO_KERNEL_TRANS_H_
