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
    this->kernel_ = new KernelTransDefault<FloatType, CoefUsage::USE_BOTH>;
  else if (static_cast<decltype(param->alpha)>(1) == param->alpha)
    this->kernel_ = new KernelTransDefault<FloatType, CoefUsage::USE_ALPHA>;
  else if (static_cast<decltype(param->beta)>(0) == param->beta)
    this->kernel_ = new KernelTransDefault<FloatType, CoefUsage::USE_BETA>;
  else
    this->kernel_ = new KernelTransDefault<FloatType, CoefUsage::USE_NONE>;

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
      this->param->input_offset, this->param->output_offset,
      this->param->alpha, this->param->beta);
}


template <typename FloatType,
          GenNumType ROWS,
          GenNumType COLS>
INLINE void kernel_tiler(DualCounter<ROWS, COLS>, GenNumType kernel_size,
    KernelTransBase<FloatType> *kernel,
    const FloatType *input_data, FloatType *output_data,
    TensorIdx input_offset, TensorIdx output_offset,
    DeducedFloatType<FloatType> alpha, DeducedFloatType<FloatType> beta) {
  // Tiling previous row
  kernel_tiler(DualCounter<ROWS - 1, COLS>(), kernel_size, kernel,
      input_data, output_data, input_offset, output_offset, alpha, beta);

  // Tiling all columns in current row
  cols_tiler(DualCounter<ROWS, COLS>(), kernel_size, kernel,
      input_data, output_data, input_offset, output_offset, alpha, beta);
}


template <typename FloatType,
          GenNumType COLS>
INLINE void kernel_tiler(DualCounter<0, COLS>, GenNumType kernel_size,
    KernelTransBase<FloatType> *kernel,
    const FloatType *input_data, FloatType *output_data,
    TensorIdx input_offset, TensorIdx output_offset,
    DeducedFloatType<FloatType> alpha, DeducedFloatType<FloatType> beta) {
  // Tiling all columns in the first row
  cols_tiler(DualCounter<0, COLS>(), kernel_size, kernel,
      input_data, output_data, input_offset, output_offset, alpha, beta);
}


template <typename FloatType,
          GenNumType ROWS,
          GenNumType COLS>
INLINE void cols_tiler(DualCounter<ROWS, COLS>, GenNumType kernel_size,
    KernelTransBase<FloatType> *kernel,
    const FloatType *input_data, FloatType *output_data,
    TensorIdx input_offset, TensorIdx output_offset,
    DeducedFloatType<FloatType> alpha, DeducedFloatType<FloatType> beta) {
  // Tiling left kernel
  cols_tiler(DualCounter<ROWS, COLS - 1>(), kernel_size, kernel,
      input_data, output_data, input_offset, output_offset, alpha, beta);

  // Tiling current kernel
  (*kernel)(input_data + COLS * kernel_size + ROWS * kernel_size * input_offset,
      output_data + ROWS * kernel_size + COLS * kernel_size * output_offset,
      input_offset, output_offset, alpha, beta);
}


template <typename FloatType,
          GenNumType ROWS>
INLINE void cols_tiler(DualCounter<ROWS, 0>, GenNumType kernel_size,
    KernelTransBase<FloatType> *kernel,
    const FloatType *input_data, FloatType *output_data,
    TensorIdx input_offset, TensorIdx output_offset,
    DeducedFloatType<FloatType> alpha, DeducedFloatType<FloatType> beta) {
  // Tiling the left-most kernel
  (*kernel)(input_data + ROWS * kernel_size * input_offset,
      output_data + ROWS * kernel_size,
      input_offset, output_offset, alpha, beta);
}

#endif // HPTC_OPERATIONS_OPERATION_TRANS_TCC_
