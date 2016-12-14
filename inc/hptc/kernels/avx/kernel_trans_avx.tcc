#pragma once
#ifndef HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_TCC_
#define HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_TCC_

template <typename FloatType, CoefUsage USAGE>
KernelTransAvxImpl<FloatType, USAGE>::KernelTransAvxImpl(
    DeducedFloatType<FloatType> alpha, DeducedFloatType<FloatType> beta) {
  this->reg_alpha = _mm256_set1_ps(static_cast<float>(alpha));
  this->reg_beta = _mm256_set1_ps(static_cast<float>(beta));
  this->regd_alpha = _mm256_set1_pd(static_cast<double>(alpha));
  this->regd_beta = _mm256_set1_pd(static_cast<double>(beta));
}


template <typename FloatType, CoefUsage USAGE>
INLINE void KernelTransAvxImpl<FloatType, USAGE>::operator()(
    const float * RESTRICT input_data, float * RESTRICT output_data,
    TensorIdx input_stride, TensorIdx output_stride) {
  // Load input data into registers
  __m256 reg_input[8], reg_output[8];
  reg_input[0] = _mm256_loadu_ps(input_data);
  reg_input[1] = _mm256_loadu_ps(input_data + input_stride);
  reg_input[2] = _mm256_loadu_ps(input_data + 2 * input_stride);
  reg_input[3] = _mm256_loadu_ps(input_data + 3 * input_stride);
  reg_input[4] = _mm256_loadu_ps(input_data + 4 * input_stride);
  reg_input[5] = _mm256_loadu_ps(input_data + 5 * input_stride);
  reg_input[6] = _mm256_loadu_ps(input_data + 6 * input_stride);
  reg_input[7] = _mm256_loadu_ps(input_data + 7 * input_stride);

  // 8x8 in-register transpose
  __m256 reg[16];
  reg[0] = _mm256_unpacklo_ps(reg_input[0], reg_input[1]);
  reg[1] = _mm256_unpackhi_ps(reg_input[0], reg_input[1]);
  reg[2] = _mm256_unpacklo_ps(reg_input[2], reg_input[3]);
  reg[3] = _mm256_unpackhi_ps(reg_input[2], reg_input[3]);
  reg[4] = _mm256_unpacklo_ps(reg_input[4], reg_input[5]);
  reg[5] = _mm256_unpackhi_ps(reg_input[4], reg_input[5]);
  reg[6] = _mm256_unpacklo_ps(reg_input[6], reg_input[7]);
  reg[7] = _mm256_unpackhi_ps(reg_input[6], reg_input[7]);

  reg[8] = _mm256_shuffle_ps(reg[0], reg[2], 0x44);
  reg[9] = _mm256_shuffle_ps(reg[0], reg[2], 0xEE);
  reg[10] = _mm256_shuffle_ps(reg[1], reg[3], 0x44);
  reg[11] = _mm256_shuffle_ps(reg[1], reg[3], 0xEE);
  reg[12] = _mm256_shuffle_ps(reg[4], reg[6], 0x44);
  reg[13] = _mm256_shuffle_ps(reg[4], reg[6], 0xEE);
  reg[14] = _mm256_shuffle_ps(reg[5], reg[7], 0x44);
  reg[15] = _mm256_shuffle_ps(reg[5], reg[7], 0xEE);

  reg_input[0] = _mm256_permute2f128_ps(reg[12], reg[8], 0x2);
  reg_input[1] = _mm256_permute2f128_ps(reg[13], reg[9], 0x2);
  reg_input[2] = _mm256_permute2f128_ps(reg[14], reg[10], 0x2);
  reg_input[3] = _mm256_permute2f128_ps(reg[15], reg[11], 0x2);
  reg_input[4] = _mm256_permute2f128_ps(reg[12], reg[8], 0x13);
  reg_input[5] = _mm256_permute2f128_ps(reg[13], reg[9], 0x13);
  reg_input[6] = _mm256_permute2f128_ps(reg[14], reg[10], 0x13);
  reg_input[7] = _mm256_permute2f128_ps(reg[15], reg[11], 0x13);

  // Rescale transposed input data
  if (CoefUsage::USE_BOTH == USAGE or CoefUsage::USE_ALPHA == USAGE) {
    reg_input[0] = _mm256_mul_ps(reg_input[0], this->reg_alpha);
    reg_input[1] = _mm256_mul_ps(reg_input[1], this->reg_alpha);
    reg_input[2] = _mm256_mul_ps(reg_input[2], this->reg_alpha);
    reg_input[3] = _mm256_mul_ps(reg_input[3], this->reg_alpha);
    reg_input[4] = _mm256_mul_ps(reg_input[4], this->reg_alpha);
    reg_input[5] = _mm256_mul_ps(reg_input[5], this->reg_alpha);
    reg_input[6] = _mm256_mul_ps(reg_input[6], this->reg_alpha);
    reg_input[7] = _mm256_mul_ps(reg_input[7], this->reg_alpha);
  }

  if (CoefUsage::USE_BOTH == USAGE or CoefUsage::USE_BETA == USAGE) {
    // Load output data into registers
    reg_output[0] = _mm256_loadu_ps(output_data);
    reg_output[1] = _mm256_loadu_ps(output_data + output_stride);
    reg_output[2] = _mm256_loadu_ps(output_data + 2 * output_stride);
    reg_output[3] = _mm256_loadu_ps(output_data + 3 * output_stride);
    reg_output[4] = _mm256_loadu_ps(output_data + 4 * output_stride);
    reg_output[5] = _mm256_loadu_ps(output_data + 5 * output_stride);
    reg_output[6] = _mm256_loadu_ps(output_data + 6 * output_stride);
    reg_output[7] = _mm256_loadu_ps(output_data + 7 * output_stride);

  // Update output data
    reg_output[0] = _mm256_mul_ps(reg_output[0], this->reg_beta);
    reg_output[1] = _mm256_mul_ps(reg_output[1], this->reg_beta);
    reg_output[2] = _mm256_mul_ps(reg_output[2], this->reg_beta);
    reg_output[3] = _mm256_mul_ps(reg_output[3], this->reg_beta);
    reg_output[4] = _mm256_mul_ps(reg_output[4], this->reg_beta);
    reg_output[5] = _mm256_mul_ps(reg_output[5], this->reg_beta);
    reg_output[6] = _mm256_mul_ps(reg_output[6], this->reg_beta);
    reg_output[7] = _mm256_mul_ps(reg_output[7], this->reg_beta);

    // Add updated result into input registers
    reg_input[0] = _mm256_add_ps(reg_output[0], reg_input[0]);
    reg_input[1] = _mm256_add_ps(reg_output[1], reg_input[1]);
    reg_input[2] = _mm256_add_ps(reg_output[2], reg_input[2]);
    reg_input[3] = _mm256_add_ps(reg_output[3], reg_input[3]);
    reg_input[4] = _mm256_add_ps(reg_output[4], reg_input[4]);
    reg_input[5] = _mm256_add_ps(reg_output[5], reg_input[5]);
    reg_input[6] = _mm256_add_ps(reg_output[6], reg_input[6]);
    reg_input[7] = _mm256_add_ps(reg_output[7], reg_input[7]);
  }

  // Write back in-register result into output data
  _mm256_storeu_ps(output_data, reg_input[0]);
  _mm256_storeu_ps(output_data + output_stride, reg_input[1]);
  _mm256_storeu_ps(output_data + 2 * output_stride, reg_input[2]);
  _mm256_storeu_ps(output_data + 3 * output_stride, reg_input[3]);
  _mm256_storeu_ps(output_data + 4 * output_stride, reg_input[4]);
  _mm256_storeu_ps(output_data + 5 * output_stride, reg_input[5]);
  _mm256_storeu_ps(output_data + 6 * output_stride, reg_input[6]);
  _mm256_storeu_ps(output_data + 7 * output_stride, reg_input[7]);
}


template <typename FloatType, CoefUsage USAGE>
INLINE void KernelTransAvxImpl<FloatType, USAGE>::operator()(
    const double * RESTRICT input_data, double * RESTRICT output_data,
    TensorIdx input_stride, TensorIdx output_stride) {
  // Load input data into registers
  __m256d reg_input[4], reg_output[4];
  reg_input[0] = _mm256_loadu_pd(input_data);
  reg_input[1] = _mm256_loadu_pd(input_data + input_stride);
  reg_input[2] = _mm256_loadu_pd(input_data + 2 * input_stride);
  reg_input[3] = _mm256_loadu_pd(input_data + 3 * input_stride);

  // 4x4 in-register transpose
  __m256d reg[4];
  reg[0] = _mm256_shuffle_pd(reg_input[0], reg_input[1], 0x3);
  reg[1] = _mm256_shuffle_pd(reg_input[0], reg_input[1], 0xC);
  reg[2] = _mm256_shuffle_pd(reg_input[2], reg_input[3], 0x3);
  reg[3] = _mm256_shuffle_pd(reg_input[2], reg_input[3], 0xC);
  reg_input[0] = _mm256_permute2f128_pd(reg[3], reg[1], 0x2);
  reg_input[1] = _mm256_permute2f128_pd(reg[2], reg[0], 0x2);
  reg_input[2] = _mm256_permute2f128_pd(reg[2], reg[0], 0x13);
  reg_input[3] = _mm256_permute2f128_pd(reg[3], reg[1], 0x13);

  // Rescale transposed input data
  if (CoefUsage::USE_BOTH == USAGE or CoefUsage::USE_ALPHA == USAGE) {
    reg_input[0] = _mm256_mul_pd(reg_input[0], this->regd_alpha);
    reg_input[1] = _mm256_mul_pd(reg_input[1], this->regd_alpha);
    reg_input[2] = _mm256_mul_pd(reg_input[2], this->regd_alpha);
    reg_input[3] = _mm256_mul_pd(reg_input[3], this->regd_alpha);
  }

  if (CoefUsage::USE_BOTH == USAGE or CoefUsage::USE_BETA == USAGE) {
    // Load output data into registers
    reg_output[0] = _mm256_loadu_pd(output_data);
    reg_output[1] = _mm256_loadu_pd(output_data + output_stride);
    reg_output[2] = _mm256_loadu_pd(output_data + 2 * output_stride);
    reg_output[3] = _mm256_loadu_pd(output_data + 3 * output_stride);

    // Update output data
    reg_output[0] = _mm256_mul_pd(reg_output[0], this->regd_beta);
    reg_output[1] = _mm256_mul_pd(reg_output[1], this->regd_beta);
    reg_output[2] = _mm256_mul_pd(reg_output[2], this->regd_beta);
    reg_output[3] = _mm256_mul_pd(reg_output[3], this->regd_beta);

    // Add updated result into input registers
    reg_input[0] = _mm256_add_pd(reg_output[0], reg_input[0]);
    reg_input[1] = _mm256_add_pd(reg_output[1], reg_input[1]);
    reg_input[2] = _mm256_add_pd(reg_output[2], reg_input[2]);
    reg_input[3] = _mm256_add_pd(reg_output[3], reg_input[3]);
  }

  // Write back in-register result into output data
  _mm256_storeu_pd(output_data, reg_input[0]);
  _mm256_storeu_pd(output_data + output_stride, reg_input[1]);
  _mm256_storeu_pd(output_data + 2 * output_stride, reg_input[2]);
  _mm256_storeu_pd(output_data + 3 * output_stride, reg_input[3]);
}


template <typename FloatType, CoefUsage USAGE>
INLINE void KernelTransAvxImpl<FloatType, USAGE>::operator()(
    const FloatComplex * RESTRICT input_data,
    FloatComplex * RESTRICT output_data,
    TensorIdx input_stride, TensorIdx output_stride) {
  // Load input data into registers
  __m256 reg_input[4], reg_output[4];
  reg_input[0] = _mm256_loadu_ps(reinterpret_cast<const float *>(input_data));
  reg_input[1] = _mm256_loadu_ps(
      reinterpret_cast<const float *>(input_data + input_stride));
  reg_input[2] = _mm256_loadu_ps(
      reinterpret_cast<const float *>(input_data + 2 * input_stride));
  reg_input[3] = _mm256_loadu_ps(
      reinterpret_cast<const float *>(input_data + 3 * input_stride));

  // 4x4 in-register transpose
  __m256 reg[4];
  reg[0] = _mm256_shuffle_ps(reg_input[0], reg_input[1], 0x44);
  reg[1] = _mm256_shuffle_ps(reg_input[0], reg_input[1], 0xEE);
  reg[2] = _mm256_shuffle_ps(reg_input[2], reg_input[3], 0x44);
  reg[3] = _mm256_shuffle_ps(reg_input[2], reg_input[3], 0xEE);
  reg_input[0] = _mm256_permute2f128_ps(reg[2], reg[0], 0x2);
  reg_input[1] = _mm256_permute2f128_ps(reg[3], reg[1], 0x2);
  reg_input[2] = _mm256_permute2f128_ps(reg[2], reg[0], 0x13);
  reg_input[3] = _mm256_permute2f128_ps(reg[3], reg[1], 0x13);

  // Rescale transposed input data
  if (CoefUsage::USE_BOTH == USAGE or CoefUsage::USE_ALPHA == USAGE) {
    reg_input[0] = _mm256_mul_ps(reg_input[0], this->reg_alpha);
    reg_input[1] = _mm256_mul_ps(reg_input[1], this->reg_alpha);
    reg_input[2] = _mm256_mul_ps(reg_input[2], this->reg_alpha);
    reg_input[3] = _mm256_mul_ps(reg_input[3], this->reg_alpha);
  }

  if (CoefUsage::USE_BOTH == USAGE or CoefUsage::USE_BETA == USAGE) {
    // Load output data into registers
    reg_output[0] = _mm256_loadu_ps(reinterpret_cast<const float *>(output_data));
    reg_output[1] = _mm256_loadu_ps(
        reinterpret_cast<const float *>(output_data + output_stride));
    reg_output[2] = _mm256_loadu_ps(
        reinterpret_cast<const float *>(output_data + 2 * output_stride));
    reg_output[3] = _mm256_loadu_ps(
        reinterpret_cast<const float *>(output_data + 3 * output_stride));

    // Update output data
    reg_output[0] = _mm256_mul_ps(reg_output[0], this->reg_beta);
    reg_output[1] = _mm256_mul_ps(reg_output[1], this->reg_beta);
    reg_output[2] = _mm256_mul_ps(reg_output[2], this->reg_beta);
    reg_output[3] = _mm256_mul_ps(reg_output[3], this->reg_beta);

    // Add updated result into input registers
    reg_input[0] = _mm256_add_ps(reg_output[0], reg_input[0]);
    reg_input[1] = _mm256_add_ps(reg_output[1], reg_input[1]);
    reg_input[2] = _mm256_add_ps(reg_output[2], reg_input[2]);
    reg_input[3] = _mm256_add_ps(reg_output[3], reg_input[3]);
  }

  // Write back in-register result into output data
  _mm256_storeu_ps(reinterpret_cast<float *>(output_data), reg_input[0]);
  _mm256_storeu_ps(reinterpret_cast<float *>(output_data + output_stride),
      reg_input[1]);
  _mm256_storeu_ps(reinterpret_cast<float *>(output_data + 2 * output_stride),
      reg_input[2]);
  _mm256_storeu_ps(reinterpret_cast<float *>(output_data + 3 * output_stride),
      reg_input[3]);
}


template <typename FloatType, CoefUsage USAGE>
INLINE void KernelTransAvxImpl<FloatType, USAGE>::operator()(
    const DoubleComplex * RESTRICT input_data,
    DoubleComplex * RESTRICT output_data,
    TensorIdx input_stride, TensorIdx output_stride) {
  // Load input data into registers
  __m256d reg_input[2], reg_output[2];
  reg_input[0] = _mm256_loadu_pd(reinterpret_cast<const double *>(input_data));
  reg_input[1] = _mm256_loadu_pd(
      reinterpret_cast<const double *>(input_data + input_stride));

  // 2x2 in-register transpose
  __m256d reg[2];
  reg[0] = _mm256_permute2f128_pd(reg_input[1], reg_input[0], 0x2);
  reg[1] = _mm256_permute2f128_pd(reg_input[1], reg_input[0], 0x13);
  reg_input[0] = reg[0];
  reg_input[1] = reg[1];

  // Rescale transposed input data
  if (CoefUsage::USE_BOTH == USAGE or CoefUsage::USE_ALPHA == USAGE) {
    reg_input[0] = _mm256_mul_pd(reg_input[0], this->regd_alpha);
    reg_input[1] = _mm256_mul_pd(reg_input[1], this->regd_alpha);
  }

  if (CoefUsage::USE_BOTH == USAGE or CoefUsage::USE_BETA == USAGE) {
    // Load output data into registers
    reg_output[0] = _mm256_loadu_pd(
        reinterpret_cast<const double *>(output_data));
    reg_output[1] = _mm256_loadu_pd(
        reinterpret_cast<const double *>(output_data + output_stride));

    // Update output data
    reg_output[0] = _mm256_mul_pd(reg_output[0], this->regd_beta);
    reg_output[1] = _mm256_mul_pd(reg_output[1], this->regd_beta);

    // Add updated result into input registers
    reg_input[0] = _mm256_add_pd(reg_output[0], reg_input[0]);
    reg_input[1] = _mm256_add_pd(reg_output[1], reg_input[1]);
  }

  // Write back in-register result into output data
  _mm256_storeu_pd(reinterpret_cast<double *>(output_data), reg_input[0]);
  _mm256_storeu_pd(reinterpret_cast<double *>(output_data + output_stride),
      reg_input[1]);
}


template <typename FloatType, CoefUsage USAGE>
INLINE GenNumType KernelTransAvxImpl<FloatType, USAGE>::get_reg_num() {
  if (std::is_same<FloatType, float>::value)
    return 8;
  else if (std::is_same<FloatType, double>::value or
      std::is_same<FloatType, FloatComplex>::value)
    return 4;
  else
    return 2;
}

#endif // HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_TCC_
