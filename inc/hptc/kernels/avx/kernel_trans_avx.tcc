#pragma once
#ifndef HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_TCC_
#define HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_TCC_

/*
 * Implementation for class KernelTransAvxBase
 */
template <>
struct KernelTransAvxBase<float, KernelType::KERNEL_FULL> {
  using RegType = __m256;

  INLINE GenNumType get_kernel_width() {
    return 8;
  }

  INLINE GenNumType get_reg_num() {
    return 8;
  }

  INLINE RegType reg_coef(float coef) {
    return _mm256_set1_ps(coef);
  }
};


template <>
struct KernelTransAvxBase<double, KernelType::KERNEL_FULL> {
  using RegType = __m256d;

  INLINE GenNumType get_kernel_width() {
    return 4;
  }

  INLINE GenNumType get_reg_num() {
    return 4;
  }

  INLINE RegType reg_coef(double coef) {
    return _mm256_set1_pd(coef);
  }
};


template <>
struct KernelTransAvxBase<FloatComplex, KernelType::KERNEL_FULL> {
  using RegType = __m256;

  INLINE GenNumType get_kernel_width() {
    return 4;
  }

  INLINE GenNumType get_reg_num() {
    return 4;
  }

  INLINE RegType reg_coef(float coef) {
    return _mm256_set1_ps(coef);
  }
};


template <>
struct KernelTransAvxBase<DoubleComplex, KernelType::KERNEL_FULL> {
  using RegType = __m256d;

  INLINE GenNumType get_kernel_width() {
    return 2;
  }

  INLINE GenNumType get_reg_num() {
    return 2;
  }

  INLINE RegType reg_coef(double coef) {
    return _mm256_set1_pd(coef);
  }
};


template <>
struct KernelTransAvxBase<float, KernelType::KERNEL_HALF> {
  using RegType = __m128;

  INLINE GenNumType get_kernel_width() {
    return 4;
  }

  INLINE GenNumType get_reg_num() {
    return 4;
  }

  INLINE RegType reg_coef(float coef) {
    return _mm_set1_ps(coef);
  }
};


template <>
struct KernelTransAvxBase<double, KernelType::KERNEL_HALF> {
  using RegType = __m128d;

  INLINE GenNumType get_kernel_width() {
    return 2;
  }

  INLINE GenNumType get_reg_num() {
    return 2;
  }

  INLINE RegType reg_coef(double coef) {
    return _mm_set1_pd(coef);
  }
};


template <>
struct KernelTransAvxBase<FloatComplex, KernelType::KERNEL_HALF> {
  using RegType = __m128;

  INLINE GenNumType get_kernel_width() {
    return 2;
  }

  INLINE GenNumType get_reg_num() {
    return 2;
  }

  INLINE RegType reg_coef(float coef) {
    return _mm_set1_ps(coef);
  }
};


template <>
struct KernelTransAvxBase<DoubleComplex, KernelType::KERNEL_HALF> {
  using RegType = double;

  INLINE GenNumType get_kernel_width() {
    return 1;
  }

  INLINE GenNumType get_reg_num() {
    return 1;
  }

  INLINE RegType reg_coef(double coef) {
    return coef;
  }
};


/*
 * Implementation for class KernelTransAvx
 */
template <CoefUsage USAGE>
struct KernelTransAvx<float, USAGE, KernelType::KERNEL_FULL> final
    : public KernelTransAvxBase<float, KernelType::KERNEL_FULL> {
  INLINE void operator()(const float * RESTRICT input_data,
      float * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride, __m256 &reg_alpha, __m256 &reg_beta) {
    // Load input data into registers
    __m256 reg_input[8];
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
    constexpr bool need_rescale = CoefUsage::USE_BOTH == USAGE or
      CoefUsage::USE_ALPHA == USAGE;
    if (need_rescale) {
      reg_input[0] = _mm256_mul_ps(reg_input[0], reg_alpha);
      reg_input[1] = _mm256_mul_ps(reg_input[1], reg_alpha);
      reg_input[2] = _mm256_mul_ps(reg_input[2], reg_alpha);
      reg_input[3] = _mm256_mul_ps(reg_input[3], reg_alpha);
      reg_input[4] = _mm256_mul_ps(reg_input[4], reg_alpha);
      reg_input[5] = _mm256_mul_ps(reg_input[5], reg_alpha);
      reg_input[6] = _mm256_mul_ps(reg_input[6], reg_alpha);
      reg_input[7] = _mm256_mul_ps(reg_input[7], reg_alpha);
    }

    constexpr bool need_update = CoefUsage::USE_BOTH == USAGE or
      CoefUsage::USE_BETA == USAGE;
    if (need_update) {
      // Load output data into registers
      __m256 reg_output[8];
      reg_output[0] = _mm256_loadu_ps(output_data);
      reg_output[1] = _mm256_loadu_ps(output_data + output_stride);
      reg_output[2] = _mm256_loadu_ps(output_data + 2 * output_stride);
      reg_output[3] = _mm256_loadu_ps(output_data + 3 * output_stride);
      reg_output[4] = _mm256_loadu_ps(output_data + 4 * output_stride);
      reg_output[5] = _mm256_loadu_ps(output_data + 5 * output_stride);
      reg_output[6] = _mm256_loadu_ps(output_data + 6 * output_stride);
      reg_output[7] = _mm256_loadu_ps(output_data + 7 * output_stride);

      // Update output data
      reg_output[0] = _mm256_mul_ps(reg_output[0], reg_beta);
      reg_output[1] = _mm256_mul_ps(reg_output[1], reg_beta);
      reg_output[2] = _mm256_mul_ps(reg_output[2], reg_beta);
      reg_output[3] = _mm256_mul_ps(reg_output[3], reg_beta);
      reg_output[4] = _mm256_mul_ps(reg_output[4], reg_beta);
      reg_output[5] = _mm256_mul_ps(reg_output[5], reg_beta);
      reg_output[6] = _mm256_mul_ps(reg_output[6], reg_beta);
      reg_output[7] = _mm256_mul_ps(reg_output[7], reg_beta);

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
};


template <CoefUsage USAGE>
struct KernelTransAvx<double, USAGE, KernelType::KERNEL_FULL> final
    : public KernelTransAvxBase<double, KernelType::KERNEL_FULL> {
  INLINE void operator()(const double * RESTRICT input_data,
      double * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride, __m256d &reg_alpha, __m256d &reg_beta) {
    // Load input data into registers
    __m256d reg_input[4];
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
    constexpr bool need_rescale = CoefUsage::USE_BOTH == USAGE or
      CoefUsage::USE_ALPHA == USAGE;
    if (need_rescale) {
      reg_input[0] = _mm256_mul_pd(reg_input[0], reg_alpha);
      reg_input[1] = _mm256_mul_pd(reg_input[1], reg_alpha);
      reg_input[2] = _mm256_mul_pd(reg_input[2], reg_alpha);
      reg_input[3] = _mm256_mul_pd(reg_input[3], reg_alpha);
    }

    constexpr bool need_update = CoefUsage::USE_BOTH == USAGE or
      CoefUsage::USE_BETA == USAGE;
    if (need_update) {
      // Load output data into registers
      __m256d reg_output[4];
      reg_output[0] = _mm256_loadu_pd(output_data);
      reg_output[1] = _mm256_loadu_pd(output_data + output_stride);
      reg_output[2] = _mm256_loadu_pd(output_data + 2 * output_stride);
      reg_output[3] = _mm256_loadu_pd(output_data + 3 * output_stride);

      // Update output data
      reg_output[0] = _mm256_mul_pd(reg_output[0], reg_beta);
      reg_output[1] = _mm256_mul_pd(reg_output[1], reg_beta);
      reg_output[2] = _mm256_mul_pd(reg_output[2], reg_beta);
      reg_output[3] = _mm256_mul_pd(reg_output[3], reg_beta);

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
};


template <CoefUsage USAGE>
struct KernelTransAvx<FloatComplex, USAGE, KernelType::KERNEL_FULL> final
    : public KernelTransAvxBase<FloatComplex, KernelType::KERNEL_FULL> {
  INLINE void operator()(const FloatComplex * RESTRICT input_data,
      FloatComplex * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride, __m256 &reg_alpha, __m256 &reg_beta) {
    // Load input data into registers
    __m256 reg_input[4];
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
    constexpr bool need_rescale = CoefUsage::USE_BOTH == USAGE or
      CoefUsage::USE_ALPHA == USAGE;
    if (need_rescale) {
      reg_input[0] = _mm256_mul_ps(reg_input[0], reg_alpha);
      reg_input[1] = _mm256_mul_ps(reg_input[1], reg_alpha);
      reg_input[2] = _mm256_mul_ps(reg_input[2], reg_alpha);
      reg_input[3] = _mm256_mul_ps(reg_input[3], reg_alpha);
    }

    constexpr bool need_update = CoefUsage::USE_BOTH == USAGE or
      CoefUsage::USE_BETA == USAGE;
    if (need_update) {
      // Load output data into registers
      __m256 reg_output[4];
      reg_output[0] = _mm256_loadu_ps(
          reinterpret_cast<const float *>(output_data));
      reg_output[1] = _mm256_loadu_ps(
          reinterpret_cast<const float *>(output_data + output_stride));
      reg_output[2] = _mm256_loadu_ps(
          reinterpret_cast<const float *>(output_data + 2 * output_stride));
      reg_output[3] = _mm256_loadu_ps(
          reinterpret_cast<const float *>(output_data + 3 * output_stride));

      // Update output data
      reg_output[0] = _mm256_mul_ps(reg_output[0], reg_beta);
      reg_output[1] = _mm256_mul_ps(reg_output[1], reg_beta);
      reg_output[2] = _mm256_mul_ps(reg_output[2], reg_beta);
      reg_output[3] = _mm256_mul_ps(reg_output[3], reg_beta);

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
};


template <CoefUsage USAGE>
struct KernelTransAvx<DoubleComplex, USAGE, KernelType::KERNEL_FULL> final
    : public KernelTransAvxBase<DoubleComplex, KernelType::KERNEL_FULL> {
  INLINE void operator()(const DoubleComplex * RESTRICT input_data,
      DoubleComplex * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride, __m256d &reg_alpha, __m256d &reg_beta) {
    // Load input data into registers
    __m256d reg_input[2];
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
    constexpr bool need_rescale = CoefUsage::USE_BOTH == USAGE or
      CoefUsage::USE_ALPHA == USAGE;
    if (need_rescale) {
      reg_input[0] = _mm256_mul_pd(reg_input[0], reg_alpha);
      reg_input[1] = _mm256_mul_pd(reg_input[1], reg_alpha);
    }

    constexpr bool need_update = CoefUsage::USE_BOTH == USAGE or
      CoefUsage::USE_BETA == USAGE;
    if (need_update) {
      // Load output data into registers
      __m256d reg_output[2];
      reg_output[0] = _mm256_loadu_pd(
          reinterpret_cast<const double *>(output_data));
      reg_output[1] = _mm256_loadu_pd(
          reinterpret_cast<const double *>(output_data + output_stride));

      // Update output data
      reg_output[0] = _mm256_mul_pd(reg_output[0], reg_beta);
      reg_output[1] = _mm256_mul_pd(reg_output[1], reg_beta);

      // Add updated result into input registers
      reg_input[0] = _mm256_add_pd(reg_output[0], reg_input[0]);
      reg_input[1] = _mm256_add_pd(reg_output[1], reg_input[1]);
    }

    // Write back in-register result into output data
    _mm256_storeu_pd(reinterpret_cast<double *>(output_data), reg_input[0]);
    _mm256_storeu_pd(reinterpret_cast<double *>(output_data + output_stride),
        reg_input[1]);
  }
};


template <CoefUsage USAGE>
struct KernelTransAvx<float, USAGE, KernelType::KERNEL_HALF> final
    : public KernelTransAvxBase<float, KernelType::KERNEL_HALF> {
  INLINE void operator()(const float * RESTRICT input_data,
      float * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride, __m128 &reg_alpha, __m128 &reg_beta) {
    // Load input data into registers
    __m128 reg_input[4];
    reg_input[0] = _mm_loadu_ps(input_data);
    reg_input[1] = _mm_loadu_ps(input_data + input_stride);
    reg_input[2] = _mm_loadu_ps(input_data + 2 * input_stride);
    reg_input[3] = _mm_loadu_ps(input_data + 3 * input_stride);

    // 4x4 in-register transpose
    __m128 reg[4];
    reg[0] = _mm_unpacklo_ps(reg_input[0], reg_input[1]);
    reg[2] = _mm_unpacklo_ps(reg_input[2], reg_input[3]);
    reg[1] = _mm_unpackhi_ps(reg_input[0], reg_input[1]);
    reg[3] = _mm_unpackhi_ps(reg_input[2], reg_input[3]);
    reg_input[0] = _mm_movelh_ps(reg[0], reg[2]);
    reg_input[1] = _mm_movehl_ps(reg[2], reg[0]);
    reg_input[2] = _mm_movelh_ps(reg[1], reg[3]);
    reg_input[3] = _mm_movehl_ps(reg[3], reg[1]);

    // Rescale transposed input_data
    constexpr bool need_rescale = CoefUsage::USE_BOTH == USAGE or
      CoefUsage::USE_ALPHA == USAGE;
    if (need_rescale) {
      reg_input[0] = _mm_mul_ps(reg_input[0], reg_alpha);
      reg_input[1] = _mm_mul_ps(reg_input[1], reg_alpha);
      reg_input[2] = _mm_mul_ps(reg_input[2], reg_alpha);
      reg_input[3] = _mm_mul_ps(reg_input[3], reg_alpha);
    }

    constexpr bool need_update = CoefUsage::USE_BOTH == USAGE or
      CoefUsage::USE_BETA == USAGE;
    if (need_update) {
      // Load output data into registers
      __m128 reg_output[4];
      reg_output[0] = _mm_loadu_ps(output_data);
      reg_output[1] = _mm_loadu_ps(output_data + output_stride);
      reg_output[2] = _mm_loadu_ps(output_data + 2 * output_stride);
      reg_output[3] = _mm_loadu_ps(output_data + 3 * output_stride);

      // Update output data
      reg_output[0] = _mm_mul_ps(reg_output[0], reg_beta);
      reg_output[1] = _mm_mul_ps(reg_output[1], reg_beta);
      reg_output[2] = _mm_mul_ps(reg_output[2], reg_beta);
      reg_output[3] = _mm_mul_ps(reg_output[3], reg_beta);

      // Add updated result into input registers
      reg_input[0] = _mm_add_ps(reg_output[0], reg_input[0]);
      reg_input[1] = _mm_add_ps(reg_output[1], reg_input[1]);
      reg_input[2] = _mm_add_ps(reg_output[2], reg_input[2]);
      reg_input[3] = _mm_add_ps(reg_output[3], reg_input[3]);
    }

    // Write back in-register result into output data
    _mm_storeu_ps(output_data, reg_input[0]);
    _mm_storeu_ps(output_data + output_stride, reg_input[1]);
    _mm_storeu_ps(output_data + 2 * output_stride, reg_input[2]);
    _mm_storeu_ps(output_data + 3 * output_stride, reg_input[3]);
  }
};


template <CoefUsage USAGE>
struct KernelTransAvx<double, USAGE, KernelType::KERNEL_HALF> final
    : public KernelTransAvxBase<double, KernelType::KERNEL_HALF> {
  INLINE void operator()(const double * RESTRICT input_data,
      double * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride, __m128d &reg_alpha, __m128d &reg_beta) {
    // Load input data into registers
    __m128d reg_input[2];
    reg_input[0] = _mm_loadu_pd(input_data);
    reg_input[1] = _mm_loadu_pd(input_data + input_stride);

    // 2x2 in-register transpose
    __m128d reg[2];
    reg[0] = _mm_unpacklo_pd(reg_input[0], reg_input[1]);
    reg[1] = _mm_unpackhi_pd(reg_input[0], reg_input[1]);

    // Rescale transposed input_data
    constexpr bool need_rescale = CoefUsage::USE_BOTH == USAGE or
      CoefUsage::USE_ALPHA == USAGE;
    if (need_rescale) {
      reg[0] = _mm_mul_pd(reg[0], reg_alpha);
      reg[1] = _mm_mul_pd(reg[1], reg_alpha);
    }

    constexpr bool need_update = CoefUsage::USE_BOTH == USAGE or
      CoefUsage::USE_BETA == USAGE;
    if (need_update) {
      // Load output data into registers
      __m128d reg_output[2];
      reg_output[0] = _mm_loadu_pd(output_data);
      reg_output[1] = _mm_loadu_pd(output_data + output_stride);

      // Update output data
      reg_output[0] = _mm_mul_pd(reg_output[0], reg_beta);
      reg_output[1] = _mm_mul_pd(reg_output[1], reg_beta);

      // Add updated result into input registers
      reg[0] = _mm_add_pd(reg_output[0], reg[0]);
      reg[1] = _mm_add_pd(reg_output[1], reg[1]);
    }

    // Write back in-register result into output data
    _mm_storeu_pd(output_data, reg[0]);
    _mm_storeu_pd(output_data + output_stride, reg[1]);
  }
};


template <CoefUsage USAGE>
struct KernelTransAvx<FloatComplex, USAGE, KernelType::KERNEL_HALF> final
    : public KernelTransAvxBase<FloatComplex, KernelType::KERNEL_HALF> {
  INLINE void operator()(const FloatComplex * RESTRICT input_data,
      FloatComplex * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride, __m128 &reg_alpha, __m128 &reg_beta) {
    // Load input data into registers
    __m128 reg_input[2];
    reg_input[0] = _mm_loadu_ps(reinterpret_cast<const float *>(input_data));
    reg_input[1] = _mm_loadu_ps(
        reinterpret_cast<const float *>(input_data + input_stride));

    // 2x2 in-register transpose
    __m128 reg[2];
    reg[0] = _mm_movelh_ps(reg_input[0], reg_input[1]);
    reg[1] = _mm_movehl_ps(reg_input[1], reg_input[0]);

    // Rescale transposed input_data
    constexpr bool need_rescale = CoefUsage::USE_BOTH == USAGE or
      CoefUsage::USE_ALPHA == USAGE;
    if (need_rescale) {
      reg[0] = _mm_mul_ps(reg[0], reg_alpha);
      reg[1] = _mm_mul_ps(reg[1], reg_alpha);
    }

    constexpr bool need_update = CoefUsage::USE_BOTH == USAGE or
      CoefUsage::USE_BETA == USAGE;
    if (need_update) {
      // Load output data into registers
      __m128 reg_output[2];
      reg_output[0] = _mm_loadu_ps(reinterpret_cast<float *>(output_data));
      reg_output[1] = _mm_loadu_ps(
          reinterpret_cast<float *>(output_data + output_stride));

      // Update output data
      reg_output[0] = _mm_mul_ps(reg_output[0], reg_beta);
      reg_output[1] = _mm_mul_ps(reg_output[1], reg_beta);

      // Add updated result into input registers
      reg[0] = _mm_add_ps(reg_output[0], reg[0]);
      reg[1] = _mm_add_ps(reg_output[1], reg[1]);
    }

    // Write back in-register result into output data
    _mm_storeu_ps(reinterpret_cast<float *>(output_data), reg[0]);
    _mm_storeu_ps(reinterpret_cast<float *>(output_data + output_stride),
        reg[1]);
  }
};


template <CoefUsage USAGE>
struct KernelTransAvx<DoubleComplex, USAGE, KernelType::KERNEL_HALF> final
    : public KernelTransAvxBase<DoubleComplex, KernelType::KERNEL_HALF> {
  INLINE void operator()(const DoubleComplex * RESTRICT input_data,
      DoubleComplex * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride, double alpha, double beta) {
    if (CoefUsage::USE_NONE == USAGE)
      *output_data = *input_data;
    else if (CoefUsage::USE_ALPHA == USAGE)
      *output_data = alpha * *input_data;
    else if (CoefUsage::USE_BETA == USAGE)
      *output_data = *input_data + beta * *output_data;
    else
      *output_data = alpha * *input_data + beta * *output_data;
  }
};

#endif // HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_TCC_
