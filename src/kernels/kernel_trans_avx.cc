#include <hptc/kernels/avx/kernel_trans_avx.h>

#include <xmmintrin.h>
#include <immintrin.h>

#include <hptc/types.h>
#include <hptc/config/config_trans.h>


namespace hptc {

/*
 * Implementation for class KernelTransAvxBase
 */
template <typename FloatType,
          KernelTypeTrans TYPE>
KernelTransAvxBase<FloatType, TYPE>::KernelTransAvxBase(Deduced coef_alpha,
    Deduced coef_beta)
    : reg_alpha(this->reg_coef(coef_alpha)),
      reg_beta(this->reg_coef(coef_beta)) {
}


template <typename FloatType,
          KernelTypeTrans TYPE>
GenNumType KernelTransAvxBase<FloatType, TYPE>::get_kernel_width() {
  constexpr GenNumType width = REG_SIZE_BYTE_AVX / sizeof(FloatType);
  if (TYPE == KernelTypeTrans::KERNEL_HALF)
    return width / 2;
  else
    return width;
}


template <typename FloatType,
          KernelTypeTrans TYPE>
GenNumType KernelTransAvxBase<FloatType, TYPE>::get_reg_num() {
  return TYPE == KernelTypeTrans::KERNEL_LINE ? 1 : this->get_kernel_width();
}


template <typename FloatType,
          KernelTypeTrans TYPE>
template <KernelTypeTrans KERNEL,
          std::enable_if_t<KERNEL == KernelTypeTrans::KERNEL_FULL or
              KERNEL == KernelTypeTrans::KERNEL_LINE> *>
DeducedRegType<float, KERNEL>
KernelTransAvxBase<FloatType, TYPE>::reg_coef(float coef) {
  return _mm256_set1_ps(coef);
}


template <typename FloatType,
          KernelTypeTrans TYPE>
template <KernelTypeTrans KERNEL,
          std::enable_if_t<KERNEL == KernelTypeTrans::KERNEL_FULL or
              KERNEL == KernelTypeTrans::KERNEL_LINE> *>
DeducedRegType<double, KERNEL>
KernelTransAvxBase<FloatType, TYPE>::reg_coef(double coef) {
  return _mm256_set1_pd(coef);
}


template <typename FloatType,
          KernelTypeTrans TYPE>
template <KernelTypeTrans KERNEL,
          std::enable_if_t<KERNEL == KernelTypeTrans::KERNEL_HALF> *>
DeducedRegType<float, KERNEL>
KernelTransAvxBase<FloatType, TYPE>::reg_coef(float coef) {
  return _mm_set1_ps(coef);
}


template <typename FloatType,
          KernelTypeTrans TYPE>
template <KernelTypeTrans KERNEL,
          std::enable_if_t<std::is_same<FloatType, double>::value and
              KERNEL == KernelTypeTrans::KERNEL_HALF> *>
DeducedRegType<double, KERNEL>
KernelTransAvxBase<FloatType, TYPE>::reg_coef(double coef) {
  return _mm_set1_pd(coef);
}


template <typename FloatType,
          KernelTypeTrans TYPE>
template <KernelTypeTrans KERNEL,
          std::enable_if_t<std::is_same<FloatType, DoubleComplex>::value and
              KERNEL == KernelTypeTrans::KERNEL_HALF> *>
DeducedRegType<DoubleComplex, KERNEL>
KernelTransAvxBase<FloatType, TYPE>::reg_coef(double coef) {
  return coef;
}


/*
 * Explicit instantiation for struct KernelTransAvx
 */
template struct KernelTransAvxBase<float, KernelTypeTrans::KERNEL_FULL>;
template struct KernelTransAvxBase<float, KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransAvxBase<float, KernelTypeTrans::KERNEL_LINE>;
template struct KernelTransAvxBase<double, KernelTypeTrans::KERNEL_FULL>;
template struct KernelTransAvxBase<double, KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransAvxBase<double, KernelTypeTrans::KERNEL_LINE>;
template struct KernelTransAvxBase<FloatComplex, KernelTypeTrans::KERNEL_FULL>;
template struct KernelTransAvxBase<FloatComplex, KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransAvxBase<FloatComplex, KernelTypeTrans::KERNEL_LINE>;
template struct KernelTransAvxBase<DoubleComplex, KernelTypeTrans::KERNEL_FULL>;
template struct KernelTransAvxBase<DoubleComplex, KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransAvxBase<DoubleComplex, KernelTypeTrans::KERNEL_LINE>;


/*
 * Implementation for class KernelTransAvx
 */
template <CoefUsageTrans USAGE>
KernelTransAvx<float, USAGE, KernelTypeTrans::KERNEL_FULL>::
KernelTransAvx(float coef_alpha, float coef_beta)
  : KernelTransAvxBase<float, KernelTypeTrans::KERNEL_FULL>(coef_alpha,
      coef_beta) {
}

template <CoefUsageTrans USAGE>
void KernelTransAvx<float, USAGE, KernelTypeTrans::KERNEL_FULL>::
operator()(const float * RESTRICT input_data,
    float * RESTRICT output_data, const TensorIdx input_stride,
    const TensorIdx output_stride) {
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
  constexpr bool need_rescale = CoefUsageTrans::USE_BOTH == USAGE or
    CoefUsageTrans::USE_ALPHA == USAGE;
  if (need_rescale) {
    reg_input[0] = _mm256_mul_ps(reg_input[0], this->reg_alpha);
    reg_input[1] = _mm256_mul_ps(reg_input[1], this->reg_alpha);
    reg_input[2] = _mm256_mul_ps(reg_input[2], this->reg_alpha);
    reg_input[3] = _mm256_mul_ps(reg_input[3], this->reg_alpha);
    reg_input[4] = _mm256_mul_ps(reg_input[4], this->reg_alpha);
    reg_input[5] = _mm256_mul_ps(reg_input[5], this->reg_alpha);
    reg_input[6] = _mm256_mul_ps(reg_input[6], this->reg_alpha);
    reg_input[7] = _mm256_mul_ps(reg_input[7], this->reg_alpha);
  }

  constexpr bool need_update = CoefUsageTrans::USE_BOTH == USAGE or
    CoefUsageTrans::USE_BETA == USAGE;
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


template <CoefUsageTrans USAGE>
KernelTransAvx<double, USAGE, KernelTypeTrans::KERNEL_FULL>::
KernelTransAvx(double coef_alpha, double coef_beta)
  : KernelTransAvxBase<double, KernelTypeTrans::KERNEL_FULL>(coef_alpha,
      coef_beta) {
}

template <CoefUsageTrans USAGE>
void KernelTransAvx<double, USAGE, KernelTypeTrans::KERNEL_FULL>::
operator()(const double * RESTRICT input_data,
    double * RESTRICT output_data, const TensorIdx input_stride,
    const TensorIdx output_stride) {
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
  constexpr bool need_rescale = CoefUsageTrans::USE_BOTH == USAGE or
    CoefUsageTrans::USE_ALPHA == USAGE;
  if (need_rescale) {
    reg_input[0] = _mm256_mul_pd(reg_input[0], this->reg_alpha);
    reg_input[1] = _mm256_mul_pd(reg_input[1], this->reg_alpha);
    reg_input[2] = _mm256_mul_pd(reg_input[2], this->reg_alpha);
    reg_input[3] = _mm256_mul_pd(reg_input[3], this->reg_alpha);
  }

  constexpr bool need_update = CoefUsageTrans::USE_BOTH == USAGE or
    CoefUsageTrans::USE_BETA == USAGE;
  if (need_update) {
    // Load output data into registers
    __m256d reg_output[4];
    reg_output[0] = _mm256_loadu_pd(output_data);
    reg_output[1] = _mm256_loadu_pd(output_data + output_stride);
    reg_output[2] = _mm256_loadu_pd(output_data + 2 * output_stride);
    reg_output[3] = _mm256_loadu_pd(output_data + 3 * output_stride);

    // Update output data
    reg_output[0] = _mm256_mul_pd(reg_output[0], this->reg_beta);
    reg_output[1] = _mm256_mul_pd(reg_output[1], this->reg_beta);
    reg_output[2] = _mm256_mul_pd(reg_output[2], this->reg_beta);
    reg_output[3] = _mm256_mul_pd(reg_output[3], this->reg_beta);

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


template <CoefUsageTrans USAGE>
KernelTransAvx<FloatComplex, USAGE, KernelTypeTrans::KERNEL_FULL>::
KernelTransAvx(float coef_alpha, float coef_beta)
  : KernelTransAvxBase<FloatComplex, KernelTypeTrans::KERNEL_FULL>(coef_alpha,
      coef_beta) {
}

template <CoefUsageTrans USAGE>
void KernelTransAvx<FloatComplex, USAGE, KernelTypeTrans::KERNEL_FULL>::
operator()(const FloatComplex * RESTRICT input_data,
    FloatComplex * RESTRICT output_data, const TensorIdx input_stride,
    const TensorIdx output_stride) {
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
  constexpr bool need_rescale = CoefUsageTrans::USE_BOTH == USAGE or
    CoefUsageTrans::USE_ALPHA == USAGE;
  if (need_rescale) {
    reg_input[0] = _mm256_mul_ps(reg_input[0], this->reg_alpha);
    reg_input[1] = _mm256_mul_ps(reg_input[1], this->reg_alpha);
    reg_input[2] = _mm256_mul_ps(reg_input[2], this->reg_alpha);
    reg_input[3] = _mm256_mul_ps(reg_input[3], this->reg_alpha);
  }

  constexpr bool need_update = CoefUsageTrans::USE_BOTH == USAGE or
    CoefUsageTrans::USE_BETA == USAGE;
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


template <CoefUsageTrans USAGE>
KernelTransAvx<DoubleComplex, USAGE, KernelTypeTrans::KERNEL_FULL>::
KernelTransAvx(double coef_alpha, double coef_beta)
  : KernelTransAvxBase<DoubleComplex, KernelTypeTrans::KERNEL_FULL>(
      coef_alpha, coef_beta) {
}

template <CoefUsageTrans USAGE>
void KernelTransAvx<DoubleComplex, USAGE, KernelTypeTrans::KERNEL_FULL>::
operator()(const DoubleComplex * RESTRICT input_data,
    DoubleComplex * RESTRICT output_data, const TensorIdx input_stride,
    const TensorIdx output_stride) {
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
  constexpr bool need_rescale = CoefUsageTrans::USE_BOTH == USAGE or
    CoefUsageTrans::USE_ALPHA == USAGE;
  if (need_rescale) {
    reg_input[0] = _mm256_mul_pd(reg_input[0], this->reg_alpha);
    reg_input[1] = _mm256_mul_pd(reg_input[1], this->reg_alpha);
  }

  constexpr bool need_update = CoefUsageTrans::USE_BOTH == USAGE or
    CoefUsageTrans::USE_BETA == USAGE;
  if (need_update) {
    // Load output data into registers
    __m256d reg_output[2];
    reg_output[0] = _mm256_loadu_pd(
        reinterpret_cast<const double *>(output_data));
    reg_output[1] = _mm256_loadu_pd(
        reinterpret_cast<const double *>(output_data + output_stride));

    // Update output data
    reg_output[0] = _mm256_mul_pd(reg_output[0], this->reg_beta);
    reg_output[1] = _mm256_mul_pd(reg_output[1], this->reg_beta);

    // Add updated result into input registers
    reg_input[0] = _mm256_add_pd(reg_output[0], reg_input[0]);
    reg_input[1] = _mm256_add_pd(reg_output[1], reg_input[1]);
  }

  // Write back in-register result into output data
  _mm256_storeu_pd(reinterpret_cast<double *>(output_data), reg_input[0]);
  _mm256_storeu_pd(reinterpret_cast<double *>(output_data + output_stride),
      reg_input[1]);
}


template <CoefUsageTrans USAGE>
KernelTransAvx<float, USAGE, KernelTypeTrans::KERNEL_HALF>::
KernelTransAvx(float coef_alpha, float coef_beta)
  : KernelTransAvxBase<float, KernelTypeTrans::KERNEL_HALF>(coef_alpha,
      coef_beta) {
}

template <CoefUsageTrans USAGE>
void KernelTransAvx<float, USAGE, KernelTypeTrans::KERNEL_HALF>::
operator()(const float * RESTRICT input_data,
    float * RESTRICT output_data, const TensorIdx input_stride,
    const TensorIdx output_stride) {
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
  constexpr bool need_rescale = CoefUsageTrans::USE_BOTH == USAGE or
    CoefUsageTrans::USE_ALPHA == USAGE;
  if (need_rescale) {
    reg_input[0] = _mm_mul_ps(reg_input[0], this->reg_alpha);
    reg_input[1] = _mm_mul_ps(reg_input[1], this->reg_alpha);
    reg_input[2] = _mm_mul_ps(reg_input[2], this->reg_alpha);
    reg_input[3] = _mm_mul_ps(reg_input[3], this->reg_alpha);
  }

  constexpr bool need_update = CoefUsageTrans::USE_BOTH == USAGE or
    CoefUsageTrans::USE_BETA == USAGE;
  if (need_update) {
    // Load output data into registers
    __m128 reg_output[4];
    reg_output[0] = _mm_loadu_ps(output_data);
    reg_output[1] = _mm_loadu_ps(output_data + output_stride);
    reg_output[2] = _mm_loadu_ps(output_data + 2 * output_stride);
    reg_output[3] = _mm_loadu_ps(output_data + 3 * output_stride);

    // Update output data
    reg_output[0] = _mm_mul_ps(reg_output[0], this->reg_beta);
    reg_output[1] = _mm_mul_ps(reg_output[1], this->reg_beta);
    reg_output[2] = _mm_mul_ps(reg_output[2], this->reg_beta);
    reg_output[3] = _mm_mul_ps(reg_output[3], this->reg_beta);

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


template <CoefUsageTrans USAGE>
KernelTransAvx<double, USAGE, KernelTypeTrans::KERNEL_HALF>::
KernelTransAvx(double coef_alpha, double coef_beta)
  : KernelTransAvxBase<double, KernelTypeTrans::KERNEL_HALF>(coef_alpha,
      coef_beta) {
}

template <CoefUsageTrans USAGE>
void KernelTransAvx<double, USAGE, KernelTypeTrans::KERNEL_HALF>::
operator()(const double * RESTRICT input_data,
    double * RESTRICT output_data, const TensorIdx input_stride,
    const TensorIdx output_stride) {
  // Load input data into registers
  __m128d reg_input[2];
  reg_input[0] = _mm_loadu_pd(input_data);
  reg_input[1] = _mm_loadu_pd(input_data + input_stride);

  // 2x2 in-register transpose
  __m128d reg[2];
  reg[0] = _mm_unpacklo_pd(reg_input[0], reg_input[1]);
  reg[1] = _mm_unpackhi_pd(reg_input[0], reg_input[1]);

  // Rescale transposed input_data
  constexpr bool need_rescale = CoefUsageTrans::USE_BOTH == USAGE or
    CoefUsageTrans::USE_ALPHA == USAGE;
  if (need_rescale) {
    reg[0] = _mm_mul_pd(reg[0], this->reg_alpha);
    reg[1] = _mm_mul_pd(reg[1], this->reg_alpha);
  }

  constexpr bool need_update = CoefUsageTrans::USE_BOTH == USAGE or
    CoefUsageTrans::USE_BETA == USAGE;
  if (need_update) {
    // Load output data into registers
    __m128d reg_output[2];
    reg_output[0] = _mm_loadu_pd(output_data);
    reg_output[1] = _mm_loadu_pd(output_data + output_stride);

    // Update output data
    reg_output[0] = _mm_mul_pd(reg_output[0], this->reg_beta);
    reg_output[1] = _mm_mul_pd(reg_output[1], this->reg_beta);

    // Add updated result into input registers
    reg[0] = _mm_add_pd(reg_output[0], reg[0]);
    reg[1] = _mm_add_pd(reg_output[1], reg[1]);
  }

  // Write back in-register result into output data
  _mm_storeu_pd(output_data, reg[0]);
  _mm_storeu_pd(output_data + output_stride, reg[1]);
}


template <CoefUsageTrans USAGE>
KernelTransAvx<FloatComplex, USAGE, KernelTypeTrans::KERNEL_HALF>::
KernelTransAvx(float coef_alpha, float coef_beta)
  : KernelTransAvxBase<FloatComplex, KernelTypeTrans::KERNEL_HALF>(coef_alpha,
      coef_beta) {
}

template <CoefUsageTrans USAGE>
void KernelTransAvx<FloatComplex, USAGE, KernelTypeTrans::KERNEL_HALF>::
operator()(const FloatComplex * RESTRICT input_data,
    FloatComplex * RESTRICT output_data, const TensorIdx input_stride,
    const TensorIdx output_stride) {
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
  constexpr bool need_rescale = CoefUsageTrans::USE_BOTH == USAGE or
    CoefUsageTrans::USE_ALPHA == USAGE;
  if (need_rescale) {
    reg[0] = _mm_mul_ps(reg[0], this->reg_alpha);
    reg[1] = _mm_mul_ps(reg[1], this->reg_alpha);
  }

  constexpr bool need_update = CoefUsageTrans::USE_BOTH == USAGE or
    CoefUsageTrans::USE_BETA == USAGE;
  if (need_update) {
    // Load output data into registers
    __m128 reg_output[2];
    reg_output[0] = _mm_loadu_ps(reinterpret_cast<float *>(output_data));
    reg_output[1] = _mm_loadu_ps(
        reinterpret_cast<float *>(output_data + output_stride));

    // Update output data
    reg_output[0] = _mm_mul_ps(reg_output[0], this->reg_beta);
    reg_output[1] = _mm_mul_ps(reg_output[1], this->reg_beta);

    // Add updated result into input registers
    reg[0] = _mm_add_ps(reg_output[0], reg[0]);
    reg[1] = _mm_add_ps(reg_output[1], reg[1]);
  }

  // Write back in-register result into output data
  _mm_storeu_ps(reinterpret_cast<float *>(output_data), reg[0]);
  _mm_storeu_ps(reinterpret_cast<float *>(output_data + output_stride),
      reg[1]);
}


template <CoefUsageTrans USAGE>
KernelTransAvx<DoubleComplex, USAGE, KernelTypeTrans::KERNEL_HALF>::
KernelTransAvx(double coef_alpha, double coef_beta)
  : KernelTransAvxBase<DoubleComplex, KernelTypeTrans::KERNEL_HALF>(
      coef_alpha, coef_beta) {
}

template <CoefUsageTrans USAGE>
void KernelTransAvx<DoubleComplex, USAGE, KernelTypeTrans::KERNEL_HALF>::
operator()(const DoubleComplex * RESTRICT input_data,
    DoubleComplex * RESTRICT output_data, const TensorIdx input_stride,
    const TensorIdx output_stride) {
  if (CoefUsageTrans::USE_NONE == USAGE)
    *output_data = *input_data;
  else if (CoefUsageTrans::USE_ALPHA == USAGE)
    *output_data = this->reg_alpha * *input_data;
  else if (CoefUsageTrans::USE_BETA == USAGE)
    *output_data = *input_data + this->reg_beta * *output_data;
  else
    *output_data = this->reg_alpha * *input_data
        + this->reg_beta * *output_data;
}


template <CoefUsageTrans USAGE>
KernelTransAvx<float, USAGE, KernelTypeTrans::KERNEL_LINE>::
KernelTransAvx(float coef_alpha, float coef_beta)
  : KernelTransAvxBase<float, KernelTypeTrans::KERNEL_LINE>(
      coef_alpha, coef_beta) {
}

template <CoefUsageTrans USAGE>
void KernelTransAvx<float, USAGE, KernelTypeTrans::KERNEL_LINE>::
operator()(const float * RESTRICT input_data,
    float * RESTRICT output_data, const TensorIdx input_stride,
    const TensorIdx output_stride) {
  // Load input data into registers
  auto reg_input = _mm256_loadu_ps(input_data);

  // Rescale transposed input data
  constexpr bool need_rescale = CoefUsageTrans::USE_BOTH == USAGE or
      CoefUsageTrans::USE_ALPHA == USAGE;
  if (need_rescale)
    reg_input = _mm256_mul_ps(reg_input, this->reg_alpha);

  // Update output data
  constexpr bool need_update = CoefUsageTrans::USE_BOTH == USAGE or
      CoefUsageTrans::USE_BETA == USAGE;
  if (need_update) {
    auto reg_output = _mm256_loadu_ps(output_data);
    reg_output = _mm256_mul_ps(reg_output, this->reg_beta);
    reg_input = _mm256_add_ps(reg_output, reg_input);
  }

  // Write data back to memory
  _mm256_storeu_ps(output_data, reg_input);
}


template <CoefUsageTrans USAGE>
KernelTransAvx<double, USAGE, KernelTypeTrans::KERNEL_LINE>::
KernelTransAvx(double coef_alpha, double coef_beta)
  : KernelTransAvxBase<double, KernelTypeTrans::KERNEL_LINE>(
      coef_alpha, coef_beta) {
}

template <CoefUsageTrans USAGE>
void KernelTransAvx<double, USAGE, KernelTypeTrans::KERNEL_LINE>::
operator()(const double * RESTRICT input_data,
    double * RESTRICT output_data, const TensorIdx input_stride,
    const TensorIdx output_stride) {
  // Load input data into registers
  auto reg_input = _mm256_loadu_pd(input_data);

  // Rescale transposed input data
  constexpr bool need_rescale = CoefUsageTrans::USE_BOTH == USAGE or
      CoefUsageTrans::USE_ALPHA == USAGE;
  if (need_rescale)
    reg_input = _mm256_mul_pd(reg_input, this->reg_alpha);

  // Update output data
  constexpr bool need_update = CoefUsageTrans::USE_BOTH == USAGE or
      CoefUsageTrans::USE_BETA == USAGE;
  if (need_update) {
    auto reg_output = _mm256_loadu_pd(output_data);
    reg_output = _mm256_mul_pd(reg_output, this->reg_beta);
    reg_input = _mm256_add_pd(reg_output, reg_input);
  }

  // Write data back to memory
  _mm256_storeu_pd(output_data, reg_input);
}


template <CoefUsageTrans USAGE>
KernelTransAvx<FloatComplex, USAGE, KernelTypeTrans::KERNEL_LINE>::
KernelTransAvx(float coef_alpha, float coef_beta)
  : KernelTransAvxBase<FloatComplex, KernelTypeTrans::KERNEL_LINE>(
      coef_alpha, coef_beta) {
}

template <CoefUsageTrans USAGE>
void KernelTransAvx<FloatComplex, USAGE, KernelTypeTrans::KERNEL_LINE>::
operator()(const FloatComplex * RESTRICT input_data,
    FloatComplex * RESTRICT output_data, const TensorIdx input_stride,
    const TensorIdx output_stride) {
  // Load input data into registers
  auto reg_input = _mm256_loadu_ps(
      reinterpret_cast<const float *>(input_data));

  // Rescale transposed input data
  constexpr bool need_rescale = CoefUsageTrans::USE_BOTH == USAGE or
      CoefUsageTrans::USE_ALPHA == USAGE;
  if (need_rescale)
    reg_input = _mm256_mul_ps(reg_input, this->reg_alpha);

  // Update output data
  constexpr bool need_update = CoefUsageTrans::USE_BOTH == USAGE or
      CoefUsageTrans::USE_BETA == USAGE;
  if (need_update) {
    auto reg_output = _mm256_loadu_ps(
        reinterpret_cast<float *>(output_data));
    reg_output = _mm256_mul_ps(reg_output, this->reg_beta);
    reg_input = _mm256_add_ps(reg_output, reg_input);
  }

  // Write data back to memory
  _mm256_storeu_ps(reinterpret_cast<float *>(output_data), reg_input);
}


template <CoefUsageTrans USAGE>
KernelTransAvx<DoubleComplex, USAGE, KernelTypeTrans::KERNEL_LINE>::
KernelTransAvx(double coef_alpha, double coef_beta)
  : KernelTransAvxBase<DoubleComplex, KernelTypeTrans::KERNEL_LINE>(
      coef_alpha, coef_beta) {
}

template <CoefUsageTrans USAGE>
void KernelTransAvx<DoubleComplex, USAGE, KernelTypeTrans::KERNEL_LINE>::
operator()(const DoubleComplex * RESTRICT input_data,
    DoubleComplex * RESTRICT output_data, const TensorIdx input_stride,
    const TensorIdx output_stride) {
  // Load input data into registers
  auto reg_input = _mm256_loadu_pd(
      reinterpret_cast<const double *>(input_data));

  // Rescale transposed input data
  constexpr bool need_rescale = CoefUsageTrans::USE_BOTH == USAGE or
      CoefUsageTrans::USE_ALPHA == USAGE;
  if (need_rescale)
    reg_input = _mm256_mul_pd(reg_input, this->reg_alpha);

  // Update output data
  constexpr bool need_update = CoefUsageTrans::USE_BOTH == USAGE or
      CoefUsageTrans::USE_BETA == USAGE;
  if (need_update) {
    auto reg_output = _mm256_loadu_pd(
        reinterpret_cast<double *>(output_data));
    reg_output = _mm256_mul_pd(reg_output, this->reg_beta);
    reg_input = _mm256_add_pd(reg_output, reg_input);
  }

  // Write data back to memory
  _mm256_storeu_pd(reinterpret_cast<double *>(output_data), reg_input);
}


/*
 * Explicit instantiation for struct KernelTransAvx
 */
template struct KernelTransAvx<float, CoefUsageTrans::USE_NONE,
         KernelTypeTrans::KERNEL_FULL>;
template struct KernelTransAvx<float, CoefUsageTrans::USE_NONE,
         KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransAvx<float, CoefUsageTrans::USE_NONE,
         KernelTypeTrans::KERNEL_LINE>;
template struct KernelTransAvx<float, CoefUsageTrans::USE_ALPHA,
         KernelTypeTrans::KERNEL_FULL>;
template struct KernelTransAvx<float, CoefUsageTrans::USE_ALPHA,
         KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransAvx<float, CoefUsageTrans::USE_ALPHA,
         KernelTypeTrans::KERNEL_LINE>;
template struct KernelTransAvx<float, CoefUsageTrans::USE_BETA,
         KernelTypeTrans::KERNEL_FULL>;
template struct KernelTransAvx<float, CoefUsageTrans::USE_BETA,
         KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransAvx<float, CoefUsageTrans::USE_BETA,
         KernelTypeTrans::KERNEL_LINE>;
template struct KernelTransAvx<float, CoefUsageTrans::USE_BOTH,
         KernelTypeTrans::KERNEL_FULL>;
template struct KernelTransAvx<float, CoefUsageTrans::USE_BOTH,
         KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransAvx<float, CoefUsageTrans::USE_BOTH,
         KernelTypeTrans::KERNEL_LINE>;
template struct KernelTransAvx<double, CoefUsageTrans::USE_NONE,
         KernelTypeTrans::KERNEL_FULL>;
template struct KernelTransAvx<double, CoefUsageTrans::USE_NONE,
         KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransAvx<double, CoefUsageTrans::USE_NONE,
         KernelTypeTrans::KERNEL_LINE>;
template struct KernelTransAvx<double, CoefUsageTrans::USE_ALPHA,
         KernelTypeTrans::KERNEL_FULL>;
template struct KernelTransAvx<double, CoefUsageTrans::USE_ALPHA,
         KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransAvx<double, CoefUsageTrans::USE_ALPHA,
         KernelTypeTrans::KERNEL_LINE>;
template struct KernelTransAvx<double, CoefUsageTrans::USE_BETA,
         KernelTypeTrans::KERNEL_FULL>;
template struct KernelTransAvx<double, CoefUsageTrans::USE_BETA,
         KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransAvx<double, CoefUsageTrans::USE_BETA,
         KernelTypeTrans::KERNEL_LINE>;
template struct KernelTransAvx<double, CoefUsageTrans::USE_BOTH,
         KernelTypeTrans::KERNEL_FULL>;
template struct KernelTransAvx<double, CoefUsageTrans::USE_BOTH,
         KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransAvx<double, CoefUsageTrans::USE_BOTH,
         KernelTypeTrans::KERNEL_LINE>;
template struct KernelTransAvx<FloatComplex, CoefUsageTrans::USE_NONE,
         KernelTypeTrans::KERNEL_FULL>;
template struct KernelTransAvx<FloatComplex, CoefUsageTrans::USE_NONE,
         KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransAvx<FloatComplex, CoefUsageTrans::USE_NONE,
         KernelTypeTrans::KERNEL_LINE>;
template struct KernelTransAvx<FloatComplex, CoefUsageTrans::USE_ALPHA,
         KernelTypeTrans::KERNEL_FULL>;
template struct KernelTransAvx<FloatComplex, CoefUsageTrans::USE_ALPHA,
         KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransAvx<FloatComplex, CoefUsageTrans::USE_ALPHA,
         KernelTypeTrans::KERNEL_LINE>;
template struct KernelTransAvx<FloatComplex, CoefUsageTrans::USE_BETA,
         KernelTypeTrans::KERNEL_FULL>;
template struct KernelTransAvx<FloatComplex, CoefUsageTrans::USE_BETA,
         KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransAvx<FloatComplex, CoefUsageTrans::USE_BETA,
         KernelTypeTrans::KERNEL_LINE>;
template struct KernelTransAvx<FloatComplex, CoefUsageTrans::USE_BOTH,
         KernelTypeTrans::KERNEL_FULL>;
template struct KernelTransAvx<FloatComplex, CoefUsageTrans::USE_BOTH,
         KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransAvx<FloatComplex, CoefUsageTrans::USE_BOTH,
         KernelTypeTrans::KERNEL_LINE>;
template struct KernelTransAvx<DoubleComplex, CoefUsageTrans::USE_NONE,
         KernelTypeTrans::KERNEL_FULL>;
template struct KernelTransAvx<DoubleComplex, CoefUsageTrans::USE_NONE,
         KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransAvx<DoubleComplex, CoefUsageTrans::USE_NONE,
         KernelTypeTrans::KERNEL_LINE>;
template struct KernelTransAvx<DoubleComplex, CoefUsageTrans::USE_ALPHA,
         KernelTypeTrans::KERNEL_FULL>;
template struct KernelTransAvx<DoubleComplex, CoefUsageTrans::USE_ALPHA,
         KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransAvx<DoubleComplex, CoefUsageTrans::USE_ALPHA,
         KernelTypeTrans::KERNEL_LINE>;
template struct KernelTransAvx<DoubleComplex, CoefUsageTrans::USE_BETA,
         KernelTypeTrans::KERNEL_FULL>;
template struct KernelTransAvx<DoubleComplex, CoefUsageTrans::USE_BETA,
         KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransAvx<DoubleComplex, CoefUsageTrans::USE_BETA,
         KernelTypeTrans::KERNEL_LINE>;
template struct KernelTransAvx<DoubleComplex, CoefUsageTrans::USE_BOTH,
         KernelTypeTrans::KERNEL_FULL>;
template struct KernelTransAvx<DoubleComplex, CoefUsageTrans::USE_BOTH,
         KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransAvx<DoubleComplex, CoefUsageTrans::USE_BOTH,
         KernelTypeTrans::KERNEL_LINE>;

}