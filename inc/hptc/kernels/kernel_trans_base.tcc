#pragma once
#ifndef HPTC_KERNELS_KERNEL_TRANS_BASE_TCC_
#define HPTC_KERNELS_KERNEL_TRANS_BASE_TCC_

template <typename FloatType>
INLINE void KernelTransBase<FloatType>::operator()(
    const FloatType * RESTRICT input_data, FloatType * RESTRICT output_data,
    TensorIdx input_offset, TensorIdx output_offset,
    DeducedFloatType<FloatType> alpha, DeducedFloatType<FloatType> beta) {
  // Read data into register and execute in-register-transpose
  this->in_reg_trans(input_data, input_offset);

  // Rescale input data if required
  this->rescale_input(alpha);

  // Update output data if required
  this->update_output(output_data, output_offset, beta);

  // Write result to memory
  this->write_back(output_data, output_offset);
}

#endif // HPTC_KERNELS_KERNEL_TRANS_BASE_TCC_
