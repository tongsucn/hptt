#pragma once
#ifndef HPTC_KERNELS_KERNEL_TRANS_BASE_TCC_
#define HPTC_KERNELS_KERNEL_TRANS_BASE_TCC_

/*template <typename FloatType>
INLINE void KernelTransBase<FloatType>::operator()(
    const FloatType * RESTRICT input_data, FloatType * RESTRICT output_data,
    TensorIdx input_stride, TensorIdx output_stride) {
  // Read data into register and execute in-register-transpose
  this->in_reg_trans(input_data, input_stride);

  // Rescale input data if required
  this->rescale_input();

  // Update output data if required
  this->update_output(output_data, output_stride);

  // Write result to memory
  this->write_back(output_data, output_stride);
}*/

#endif // HPTC_KERNELS_KERNEL_TRANS_BASE_TCC_
