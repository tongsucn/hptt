#pragma once
#ifndef HPTC_OPERATIONS_OPERATION_TRANS_TCC_
#define HPTC_OPERATIONS_OPERATION_TRANS_TCC_

/*
 * Implementation for class OpMacroTrans
 */
template <typename FloatType,
          GenNumType HEIGHT,
          GenNumType WIDTH>
OpMacroTrans<FloatType, HEIGHT, WIDTH>::OpMacroTrans(
    const std::shared_ptr<ParamTrans<FloatType>> &param)
    : OpMacro<FloatType, ParamTrans>(param) {
  // Check coefficient and create correspondence kernel
  if (static_cast<decltype(param->alpha)>(1) == param->alpha
      and static_cast<decltype(param->beta)>(0) == param->beta)
    this->kernel_ = new KernelTrans<FloatType, CoefUsage::USE_NONE>(
        param->alpha, param->beta);
  else if (static_cast<decltype(param->alpha)>(1) == param->alpha)
    this->kernel_ = new KernelTrans<FloatType, CoefUsage::USE_BETA>(
        param->alpha, param->beta);
  else if (static_cast<decltype(param->beta)>(0) == param->beta)
    this->kernel_ = new KernelTrans<FloatType, CoefUsage::USE_ALPHA>(
        param->alpha, param->beta);
  else
    this->kernel_ = new KernelTrans<FloatType, CoefUsage::USE_BOTH>(
        param->alpha, param->beta);

  this->kernel_size_ = this->kernel_->get_reg_num();
}


template <typename FloatType,
          GenNumType HEIGHT,
          GenNumType WIDTH>
OpMacroTrans<FloatType, HEIGHT, WIDTH>::~OpMacroTrans() {
  delete kernel_;
}


template <typename FloatType,
          GenNumType HEIGHT,
          GenNumType WIDTH>
INLINE void OpMacroTrans<FloatType, HEIGHT, WIDTH>::exec() {
  kernel_tiler(DualCounter<HEIGHT - 1, WIDTH - 1>(), this->kernel_size_,
      this->kernel_, &this->param->input_tensor[this->param->macro_loop_idx],
      &this->param->output_tensor[this->param->macro_loop_perm_idx],
      this->param->input_stride, this->param->output_stride);
}


template <typename FloatType,
          GenNumType ROWS,
          GenNumType COLS>
INLINE void kernel_tiler(DualCounter<ROWS, COLS>, GenNumType kernel_size,
    KernelTransBase<FloatType> *kernel,
    const FloatType *input_data, FloatType *output_data,
    TensorIdx input_stride, TensorIdx output_stride) {
  // Tiling previous row
  kernel_tiler(DualCounter<ROWS - 1, COLS>(), kernel_size, kernel,
      input_data, output_data, input_stride, output_stride);

  // Tiling all columns in current row
  cols_tiler(DualCounter<ROWS, COLS>(), kernel_size, kernel,
      input_data, output_data, input_stride, output_stride);
}


template <typename FloatType,
          GenNumType COLS>
INLINE void kernel_tiler(DualCounter<0, COLS>, GenNumType kernel_size,
    KernelTransBase<FloatType> *kernel,
    const FloatType *input_data, FloatType *output_data,
    TensorIdx input_stride, TensorIdx output_stride) {
  // Tiling all columns in the first row
  cols_tiler(DualCounter<0, COLS>(), kernel_size, kernel,
      input_data, output_data, input_stride, output_stride);
}


template <typename FloatType,
          GenNumType ROWS,
          GenNumType COLS>
INLINE void cols_tiler(DualCounter<ROWS, COLS>, GenNumType kernel_size,
    KernelTransBase<FloatType> *kernel,
    const FloatType *input_data, FloatType *output_data,
    TensorIdx input_stride, TensorIdx output_stride) {
  // Tiling left kernel
  cols_tiler(DualCounter<ROWS, COLS - 1>(), kernel_size, kernel,
      input_data, output_data, input_stride, output_stride);

  // Tiling current kernel
  (*kernel)(input_data + COLS * kernel_size + ROWS * kernel_size * input_stride,
      output_data + ROWS * kernel_size + COLS * kernel_size * output_stride,
      input_stride, output_stride);
}


template <typename FloatType,
          GenNumType ROWS>
INLINE void cols_tiler(DualCounter<ROWS, 0>, GenNumType kernel_size,
    KernelTransBase<FloatType> *kernel,
    const FloatType *input_data, FloatType *output_data,
    TensorIdx input_stride, TensorIdx output_stride) {
  // Tiling the left-most kernel
  (*kernel)(input_data + ROWS * kernel_size * input_stride,
      output_data + ROWS * kernel_size, input_stride, output_stride);
}

#endif // HPTC_OPERATIONS_OPERATION_TRANS_TCC_
