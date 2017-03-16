#include <hptc/kernels/avx/kernel_trans_avx.h>

#include <xmmintrin.h>
#include <immintrin.h>

#include <hptc/types.h>
#include <hptc/config/config_trans.h>


namespace hptc {

/*
 * Implementation for class KernelTransAvx
 */
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
    const TensorIdx output_stride) const {
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
    const TensorIdx output_stride) const {
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
    const TensorIdx output_stride) const {
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
    const TensorIdx output_stride) const {
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


/*
 * Explicit instantiation for struct KernelTransAvx
 */
template struct KernelTransAvx<float, CoefUsageTrans::USE_NONE,
         KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransAvx<float, CoefUsageTrans::USE_ALPHA,
         KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransAvx<float, CoefUsageTrans::USE_BETA,
         KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransAvx<float, CoefUsageTrans::USE_BOTH,
         KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransAvx<double, CoefUsageTrans::USE_NONE,
         KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransAvx<double, CoefUsageTrans::USE_ALPHA,
         KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransAvx<double, CoefUsageTrans::USE_BETA,
         KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransAvx<double, CoefUsageTrans::USE_BOTH,
         KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransAvx<FloatComplex, CoefUsageTrans::USE_NONE,
         KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransAvx<FloatComplex, CoefUsageTrans::USE_ALPHA,
         KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransAvx<FloatComplex, CoefUsageTrans::USE_BETA,
         KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransAvx<FloatComplex, CoefUsageTrans::USE_BOTH,
         KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransAvx<DoubleComplex, CoefUsageTrans::USE_NONE,
         KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransAvx<DoubleComplex, CoefUsageTrans::USE_ALPHA,
         KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransAvx<DoubleComplex, CoefUsageTrans::USE_BETA,
         KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransAvx<DoubleComplex, CoefUsageTrans::USE_BOTH,
         KernelTypeTrans::KERNEL_HALF>;

}
