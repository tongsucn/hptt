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
KernelTransAvx<float, USAGE, KernelTypeTrans::KERNEL_LINE>::
KernelTransAvx(float coef_alpha, float coef_beta)
  : KernelTransAvxBase<float, KernelTypeTrans::KERNEL_LINE>(coef_alpha,
      coef_beta) {
}

template <CoefUsageTrans USAGE>
void KernelTransAvx<float, USAGE, KernelTypeTrans::KERNEL_LINE>::
operator()(const float * RESTRICT input_data,
    float * RESTRICT output_data, const TensorIdx input_stride,
    const TensorIdx output_stride) const {
  // Load input data into registers
  __m256 reg_input;
  reg_input = _mm256_loadu_ps(input_data);

  // Rescale transposed input data
  constexpr bool need_rescale = CoefUsageTrans::USE_BOTH == USAGE or
    CoefUsageTrans::USE_ALPHA == USAGE;
  if (need_rescale)
    reg_input = _mm256_mul_ps(reg_input, this->reg_alpha);

  constexpr bool need_update = CoefUsageTrans::USE_BOTH == USAGE or
    CoefUsageTrans::USE_BETA == USAGE;
  if (need_update) {
    // Load output data into registers
    __m256 reg_output;
    reg_output = _mm256_loadu_ps(output_data);
    // Update output data
    reg_output = _mm256_mul_ps(reg_output, this->reg_beta);
    // Add updated result into input registers
    reg_input = _mm256_add_ps(reg_output, reg_input);
  }

  // Write back in-register result into output data
  _mm256_storeu_ps(output_data, reg_input);
}


template <CoefUsageTrans USAGE>
KernelTransAvx<double, USAGE, KernelTypeTrans::KERNEL_LINE>::
KernelTransAvx(double coef_alpha, double coef_beta)
  : KernelTransAvxBase<double, KernelTypeTrans::KERNEL_LINE>(coef_alpha,
      coef_beta) {
}

template <CoefUsageTrans USAGE>
void KernelTransAvx<double, USAGE, KernelTypeTrans::KERNEL_LINE>::
operator()(const double * RESTRICT input_data,
    double * RESTRICT output_data, const TensorIdx input_stride,
    const TensorIdx output_stride) const {
  // Load input data into registers
  __m256d reg_input;
  reg_input = _mm256_loadu_pd(input_data);

  // Rescale transposed input data
  constexpr bool need_rescale = CoefUsageTrans::USE_BOTH == USAGE or
    CoefUsageTrans::USE_ALPHA == USAGE;
  if (need_rescale)
    reg_input = _mm256_mul_pd(reg_input, this->reg_alpha);

  constexpr bool need_update = CoefUsageTrans::USE_BOTH == USAGE or
    CoefUsageTrans::USE_BETA == USAGE;
  if (need_update) {
    // Load output data into registers
    __m256d reg_output;
    reg_output = _mm256_loadu_pd(output_data);
    // Update output data
    reg_output = _mm256_mul_pd(reg_output, this->reg_beta);
    // Add updated result into input registers
    reg_input = _mm256_add_pd(reg_output, reg_input);
  }

  // Write back in-register result into output data
  _mm256_storeu_pd(output_data, reg_input);
}


template <CoefUsageTrans USAGE>
KernelTransAvx<FloatComplex, USAGE, KernelTypeTrans::KERNEL_LINE>::
KernelTransAvx(float coef_alpha, float coef_beta)
  : KernelTransAvxBase<FloatComplex, KernelTypeTrans::KERNEL_LINE>(coef_alpha,
      coef_beta) {
}

template <CoefUsageTrans USAGE>
void KernelTransAvx<FloatComplex, USAGE, KernelTypeTrans::KERNEL_LINE>::
operator()(const FloatComplex * RESTRICT input_data,
    FloatComplex * RESTRICT output_data, const TensorIdx input_stride,
    const TensorIdx output_stride) const {
  // Load input data into registers
  __m256 reg_input;
  reg_input = _mm256_loadu_ps(reinterpret_cast<const float *>(input_data));

  // Rescale transposed input data
  constexpr bool need_rescale = CoefUsageTrans::USE_BOTH == USAGE or
    CoefUsageTrans::USE_ALPHA == USAGE;
  if (need_rescale)
    reg_input = _mm256_mul_ps(reg_input, this->reg_alpha);

  constexpr bool need_update = CoefUsageTrans::USE_BOTH == USAGE or
    CoefUsageTrans::USE_BETA == USAGE;
  if (need_update) {
    // Load output data into registers
    __m256 reg_output;
    reg_output = _mm256_loadu_ps(reinterpret_cast<float *>(output_data));
    // Update output data
    reg_output = _mm256_mul_ps(reg_output, this->reg_beta);
    // Add updated result into input registers
    reg_input = _mm256_add_ps(reg_output, reg_input);
  }

  // Write back in-register result into output data
  _mm256_storeu_ps(reinterpret_cast<float *>(output_data), reg_input);
}


template <CoefUsageTrans USAGE>
KernelTransAvx<DoubleComplex, USAGE, KernelTypeTrans::KERNEL_LINE>::
KernelTransAvx(double coef_alpha, double coef_beta)
  : KernelTransAvxBase<DoubleComplex, KernelTypeTrans::KERNEL_LINE>(coef_alpha,
      coef_beta) {
}

template <CoefUsageTrans USAGE>
void KernelTransAvx<DoubleComplex, USAGE, KernelTypeTrans::KERNEL_LINE>::
operator()(const DoubleComplex * RESTRICT input_data,
    DoubleComplex * RESTRICT output_data, const TensorIdx input_stride,
    const TensorIdx output_stride) const {
  // Load input data into registers
  __m256d reg_input;
  reg_input = _mm256_loadu_pd(reinterpret_cast<const double *>(input_data));

  // Rescale transposed input data
  constexpr bool need_rescale = CoefUsageTrans::USE_BOTH == USAGE or
    CoefUsageTrans::USE_ALPHA == USAGE;
  if (need_rescale)
    reg_input = _mm256_mul_pd(reg_input, this->reg_alpha);

  constexpr bool need_update = CoefUsageTrans::USE_BOTH == USAGE or
    CoefUsageTrans::USE_BETA == USAGE;
  if (need_update) {
    // Load output data into registers
    __m256d reg_output;
    reg_output = _mm256_loadu_pd(reinterpret_cast<double *>(output_data));
    // Update output data
    reg_output = _mm256_mul_pd(reg_output, this->reg_beta);
    // Add updated result into input registers
    reg_input = _mm256_add_pd(reg_output, reg_input);
  }

  // Write back in-register result into output data
  _mm256_storeu_pd(reinterpret_cast<double *>(output_data), reg_input);
}


/*
 * Explicit instantiation for struct KernelTransAvx
 */
template struct KernelTransAvx<float, CoefUsageTrans::USE_ALPHA,
         KernelTypeTrans::KERNEL_LINE>;
template struct KernelTransAvx<float, CoefUsageTrans::USE_BETA,
         KernelTypeTrans::KERNEL_LINE>;
template struct KernelTransAvx<float, CoefUsageTrans::USE_BOTH,
         KernelTypeTrans::KERNEL_LINE>;
template struct KernelTransAvx<double, CoefUsageTrans::USE_ALPHA,
         KernelTypeTrans::KERNEL_LINE>;
template struct KernelTransAvx<double, CoefUsageTrans::USE_BETA,
         KernelTypeTrans::KERNEL_LINE>;
template struct KernelTransAvx<double, CoefUsageTrans::USE_BOTH,
         KernelTypeTrans::KERNEL_LINE>;
template struct KernelTransAvx<FloatComplex, CoefUsageTrans::USE_ALPHA,
         KernelTypeTrans::KERNEL_LINE>;
template struct KernelTransAvx<FloatComplex, CoefUsageTrans::USE_BETA,
         KernelTypeTrans::KERNEL_LINE>;
template struct KernelTransAvx<FloatComplex, CoefUsageTrans::USE_BOTH,
         KernelTypeTrans::KERNEL_LINE>;
template struct KernelTransAvx<DoubleComplex, CoefUsageTrans::USE_ALPHA,
         KernelTypeTrans::KERNEL_LINE>;
template struct KernelTransAvx<DoubleComplex, CoefUsageTrans::USE_BETA,
         KernelTypeTrans::KERNEL_LINE>;
template struct KernelTransAvx<DoubleComplex, CoefUsageTrans::USE_BOTH,
         KernelTypeTrans::KERNEL_LINE>;

}
