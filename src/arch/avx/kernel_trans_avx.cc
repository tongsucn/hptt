#include <hptc/arch/avx/kernel_trans_avx.h>

#include <algorithm>

#include <immintrin.h>
#include <xmmintrin.h>

#include <hptc/types.h>
#include <hptc/arch/compat.h>
#include <hptc/util/util_trans.h>


namespace hptc {

template <>
void KernelTrans<float, KernelTypeTrans::KERNEL_FULL>::exec(
    const float * RESTRICT in_data, float * RESTRICT out_data,
    const TensorIdx input_stride, const TensorIdx output_stride) const {
  // Load input data into registers
  __m256 reg_input[8];
  reg_input[0] = _mm256_loadu_ps(in_data);
  reg_input[1] = _mm256_loadu_ps(in_data + input_stride);
  reg_input[2] = _mm256_loadu_ps(in_data + 2 * input_stride);
  reg_input[3] = _mm256_loadu_ps(in_data + 3 * input_stride);
  reg_input[4] = _mm256_loadu_ps(in_data + 4 * input_stride);
  reg_input[5] = _mm256_loadu_ps(in_data + 5 * input_stride);
  reg_input[6] = _mm256_loadu_ps(in_data + 6 * input_stride);
  reg_input[7] = _mm256_loadu_ps(in_data + 7 * input_stride);

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
  reg_input[0] = _mm256_mul_ps(reg_input[0], this->reg_alpha_);
  reg_input[1] = _mm256_mul_ps(reg_input[1], this->reg_alpha_);
  reg_input[2] = _mm256_mul_ps(reg_input[2], this->reg_alpha_);
  reg_input[3] = _mm256_mul_ps(reg_input[3], this->reg_alpha_);
  reg_input[4] = _mm256_mul_ps(reg_input[4], this->reg_alpha_);
  reg_input[5] = _mm256_mul_ps(reg_input[5], this->reg_alpha_);
  reg_input[6] = _mm256_mul_ps(reg_input[6], this->reg_alpha_);
  reg_input[7] = _mm256_mul_ps(reg_input[7], this->reg_alpha_);

  // Load output data into registers
  __m256 reg_output[8];
  reg_output[0] = _mm256_loadu_ps(out_data);
  reg_output[1] = _mm256_loadu_ps(out_data + output_stride);
  reg_output[2] = _mm256_loadu_ps(out_data + 2 * output_stride);
  reg_output[3] = _mm256_loadu_ps(out_data + 3 * output_stride);
  reg_output[4] = _mm256_loadu_ps(out_data + 4 * output_stride);
  reg_output[5] = _mm256_loadu_ps(out_data + 5 * output_stride);
  reg_output[6] = _mm256_loadu_ps(out_data + 6 * output_stride);
  reg_output[7] = _mm256_loadu_ps(out_data + 7 * output_stride);

  // Update output data
  reg_output[0] = _mm256_mul_ps(reg_output[0], this->reg_beta_);
  reg_output[1] = _mm256_mul_ps(reg_output[1], this->reg_beta_);
  reg_output[2] = _mm256_mul_ps(reg_output[2], this->reg_beta_);
  reg_output[3] = _mm256_mul_ps(reg_output[3], this->reg_beta_);
  reg_output[4] = _mm256_mul_ps(reg_output[4], this->reg_beta_);
  reg_output[5] = _mm256_mul_ps(reg_output[5], this->reg_beta_);
  reg_output[6] = _mm256_mul_ps(reg_output[6], this->reg_beta_);
  reg_output[7] = _mm256_mul_ps(reg_output[7], this->reg_beta_);

  // Add updated result into input registers
  reg_input[0] = _mm256_add_ps(reg_output[0], reg_input[0]);
  reg_input[1] = _mm256_add_ps(reg_output[1], reg_input[1]);
  reg_input[2] = _mm256_add_ps(reg_output[2], reg_input[2]);
  reg_input[3] = _mm256_add_ps(reg_output[3], reg_input[3]);
  reg_input[4] = _mm256_add_ps(reg_output[4], reg_input[4]);
  reg_input[5] = _mm256_add_ps(reg_output[5], reg_input[5]);
  reg_input[6] = _mm256_add_ps(reg_output[6], reg_input[6]);
  reg_input[7] = _mm256_add_ps(reg_output[7], reg_input[7]);

  // Write back in-register result into output data
  _mm256_storeu_ps(out_data, reg_input[0]);
  _mm256_storeu_ps(out_data + output_stride, reg_input[1]);
  _mm256_storeu_ps(out_data + 2 * output_stride, reg_input[2]);
  _mm256_storeu_ps(out_data + 3 * output_stride, reg_input[3]);
  _mm256_storeu_ps(out_data + 4 * output_stride, reg_input[4]);
  _mm256_storeu_ps(out_data + 5 * output_stride, reg_input[5]);
  _mm256_storeu_ps(out_data + 6 * output_stride, reg_input[6]);
  _mm256_storeu_ps(out_data + 7 * output_stride, reg_input[7]);
}


template <>
void KernelTrans<double, KernelTypeTrans::KERNEL_FULL>::exec(
    const double * RESTRICT in_data, double * RESTRICT out_data,
    const TensorIdx input_stride, const TensorIdx output_stride) const {
  // Load input data into registers
  __m256d reg_input[4];
  reg_input[0] = _mm256_loadu_pd(in_data);
  reg_input[1] = _mm256_loadu_pd(in_data + input_stride);
  reg_input[2] = _mm256_loadu_pd(in_data + 2 * input_stride);
  reg_input[3] = _mm256_loadu_pd(in_data + 3 * input_stride);

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
  reg_input[0] = _mm256_mul_pd(reg_input[0], this->reg_alpha_);
  reg_input[1] = _mm256_mul_pd(reg_input[1], this->reg_alpha_);
  reg_input[2] = _mm256_mul_pd(reg_input[2], this->reg_alpha_);
  reg_input[3] = _mm256_mul_pd(reg_input[3], this->reg_alpha_);

  // Load output data into registers
  __m256d reg_output[4];
  reg_output[0] = _mm256_loadu_pd(out_data);
  reg_output[1] = _mm256_loadu_pd(out_data + output_stride);
  reg_output[2] = _mm256_loadu_pd(out_data + 2 * output_stride);
  reg_output[3] = _mm256_loadu_pd(out_data + 3 * output_stride);

  // Update output data
  reg_output[0] = _mm256_mul_pd(reg_output[0], this->reg_beta_);
  reg_output[1] = _mm256_mul_pd(reg_output[1], this->reg_beta_);
  reg_output[2] = _mm256_mul_pd(reg_output[2], this->reg_beta_);
  reg_output[3] = _mm256_mul_pd(reg_output[3], this->reg_beta_);

  // Add updated result into input registers
  reg_input[0] = _mm256_add_pd(reg_output[0], reg_input[0]);
  reg_input[1] = _mm256_add_pd(reg_output[1], reg_input[1]);
  reg_input[2] = _mm256_add_pd(reg_output[2], reg_input[2]);
  reg_input[3] = _mm256_add_pd(reg_output[3], reg_input[3]);

  // Write back in-register result into output data
  _mm256_storeu_pd(out_data, reg_input[0]);
  _mm256_storeu_pd(out_data + output_stride, reg_input[1]);
  _mm256_storeu_pd(out_data + 2 * output_stride, reg_input[2]);
  _mm256_storeu_pd(out_data + 3 * output_stride, reg_input[3]);
}


template <>
void KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_FULL>::exec(
    const FloatComplex * RESTRICT in_data, FloatComplex * RESTRICT out_data,
    const TensorIdx input_stride, const TensorIdx output_stride) const {
  // Load input data into registers
  __m256 reg_input[4];
  reg_input[0] = _mm256_loadu_ps(reinterpret_cast<const float *>(in_data));
  reg_input[1] = _mm256_loadu_ps(
      reinterpret_cast<const float *>(in_data + input_stride));
  reg_input[2] = _mm256_loadu_ps(
      reinterpret_cast<const float *>(in_data + 2 * input_stride));
  reg_input[3] = _mm256_loadu_ps(
      reinterpret_cast<const float *>(in_data + 3 * input_stride));

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
  reg_input[0] = _mm256_mul_ps(reg_input[0], this->reg_alpha_);
  reg_input[1] = _mm256_mul_ps(reg_input[1], this->reg_alpha_);
  reg_input[2] = _mm256_mul_ps(reg_input[2], this->reg_alpha_);
  reg_input[3] = _mm256_mul_ps(reg_input[3], this->reg_alpha_);

  // Load output data into registers
  __m256 reg_output[4];
  reg_output[0] = _mm256_loadu_ps(
      reinterpret_cast<const float *>(out_data));
  reg_output[1] = _mm256_loadu_ps(
      reinterpret_cast<const float *>(out_data + output_stride));
  reg_output[2] = _mm256_loadu_ps(
      reinterpret_cast<const float *>(out_data + 2 * output_stride));
  reg_output[3] = _mm256_loadu_ps(
      reinterpret_cast<const float *>(out_data + 3 * output_stride));

  // Update output data
  reg_output[0] = _mm256_mul_ps(reg_output[0], this->reg_beta_);
  reg_output[1] = _mm256_mul_ps(reg_output[1], this->reg_beta_);
  reg_output[2] = _mm256_mul_ps(reg_output[2], this->reg_beta_);
  reg_output[3] = _mm256_mul_ps(reg_output[3], this->reg_beta_);

  // Add updated result into input registers
  reg_input[0] = _mm256_add_ps(reg_output[0], reg_input[0]);
  reg_input[1] = _mm256_add_ps(reg_output[1], reg_input[1]);
  reg_input[2] = _mm256_add_ps(reg_output[2], reg_input[2]);
  reg_input[3] = _mm256_add_ps(reg_output[3], reg_input[3]);

  // Write back in-register result into output data
  _mm256_storeu_ps(reinterpret_cast<float *>(out_data), reg_input[0]);
  _mm256_storeu_ps(reinterpret_cast<float *>(out_data + output_stride),
      reg_input[1]);
  _mm256_storeu_ps(reinterpret_cast<float *>(out_data + 2 * output_stride),
      reg_input[2]);
  _mm256_storeu_ps(reinterpret_cast<float *>(out_data + 3 * output_stride),
      reg_input[3]);
}


template <>
void KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_FULL>::exec(
    const DoubleComplex * RESTRICT in_data, DoubleComplex * RESTRICT out_data,
    const TensorIdx input_stride, const TensorIdx output_stride) const {
  // Load input data into registers
  __m256d reg_input[2];
  reg_input[0] = _mm256_loadu_pd(reinterpret_cast<const double *>(in_data));
  reg_input[1] = _mm256_loadu_pd(
      reinterpret_cast<const double *>(in_data + input_stride));

  // 2x2 in-register transpose
  __m256d reg[2];
  reg[0] = _mm256_permute2f128_pd(reg_input[1], reg_input[0], 0x2);
  reg[1] = _mm256_permute2f128_pd(reg_input[1], reg_input[0], 0x13);
  reg_input[0] = reg[0];
  reg_input[1] = reg[1];

  // Rescale transposed input data
  reg_input[0] = _mm256_mul_pd(reg_input[0], this->reg_alpha_);
  reg_input[1] = _mm256_mul_pd(reg_input[1], this->reg_alpha_);

  // Load output data into registers
  __m256d reg_output[2];
  reg_output[0] = _mm256_loadu_pd(
      reinterpret_cast<const double *>(out_data));
  reg_output[1] = _mm256_loadu_pd(
      reinterpret_cast<const double *>(out_data + output_stride));

  // Update output data
  reg_output[0] = _mm256_mul_pd(reg_output[0], this->reg_beta_);
  reg_output[1] = _mm256_mul_pd(reg_output[1], this->reg_beta_);

  // Add updated result into input registers
  reg_input[0] = _mm256_add_pd(reg_output[0], reg_input[0]);
  reg_input[1] = _mm256_add_pd(reg_output[1], reg_input[1]);

  // Write back in-register result into output data
  _mm256_storeu_pd(reinterpret_cast<double *>(out_data), reg_input[0]);
  _mm256_storeu_pd(reinterpret_cast<double *>(out_data + output_stride),
      reg_input[1]);
}


template <>
void KernelTrans<float, KernelTypeTrans::KERNEL_HALF>::exec(
    const float * RESTRICT in_data, float * RESTRICT out_data,
    const TensorIdx input_stride, const TensorIdx output_stride) const {
  // Load input data into registers
  __m128 reg_input[4];
  reg_input[0] = _mm_loadu_ps(in_data);
  reg_input[1] = _mm_loadu_ps(in_data + input_stride);
  reg_input[2] = _mm_loadu_ps(in_data + 2 * input_stride);
  reg_input[3] = _mm_loadu_ps(in_data + 3 * input_stride);

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

  // Rescale transposed in_data
  reg_input[0] = _mm_mul_ps(reg_input[0], this->reg_alpha_);
  reg_input[1] = _mm_mul_ps(reg_input[1], this->reg_alpha_);
  reg_input[2] = _mm_mul_ps(reg_input[2], this->reg_alpha_);
  reg_input[3] = _mm_mul_ps(reg_input[3], this->reg_alpha_);

  // Load output data into registers
  __m128 reg_output[4];
  reg_output[0] = _mm_loadu_ps(out_data);
  reg_output[1] = _mm_loadu_ps(out_data + output_stride);
  reg_output[2] = _mm_loadu_ps(out_data + 2 * output_stride);
  reg_output[3] = _mm_loadu_ps(out_data + 3 * output_stride);

  // Update output data
  reg_output[0] = _mm_mul_ps(reg_output[0], this->reg_beta_);
  reg_output[1] = _mm_mul_ps(reg_output[1], this->reg_beta_);
  reg_output[2] = _mm_mul_ps(reg_output[2], this->reg_beta_);
  reg_output[3] = _mm_mul_ps(reg_output[3], this->reg_beta_);

  // Add updated result into input registers
  reg_input[0] = _mm_add_ps(reg_output[0], reg_input[0]);
  reg_input[1] = _mm_add_ps(reg_output[1], reg_input[1]);
  reg_input[2] = _mm_add_ps(reg_output[2], reg_input[2]);
  reg_input[3] = _mm_add_ps(reg_output[3], reg_input[3]);

  // Write back in-register result into output data
  _mm_storeu_ps(out_data, reg_input[0]);
  _mm_storeu_ps(out_data + output_stride, reg_input[1]);
  _mm_storeu_ps(out_data + 2 * output_stride, reg_input[2]);
  _mm_storeu_ps(out_data + 3 * output_stride, reg_input[3]);
}


template <>
void KernelTrans<double, KernelTypeTrans::KERNEL_HALF>::exec(
    const double * RESTRICT in_data, double * RESTRICT out_data,
    const TensorIdx input_stride, const TensorIdx output_stride) const {
  // Load input data into registers
  __m128d reg_input[2];
  reg_input[0] = _mm_loadu_pd(in_data);
  reg_input[1] = _mm_loadu_pd(in_data + input_stride);

  // 2x2 in-register transpose
  __m128d reg[2];
  reg[0] = _mm_unpacklo_pd(reg_input[0], reg_input[1]);
  reg[1] = _mm_unpackhi_pd(reg_input[0], reg_input[1]);

  // Rescale transposed in_data
  reg[0] = _mm_mul_pd(reg[0], this->reg_alpha_);
  reg[1] = _mm_mul_pd(reg[1], this->reg_alpha_);

  // Load output data into registers
  __m128d reg_output[2];
  reg_output[0] = _mm_loadu_pd(out_data);
  reg_output[1] = _mm_loadu_pd(out_data + output_stride);

  // Update output data
  reg_output[0] = _mm_mul_pd(reg_output[0], this->reg_beta_);
  reg_output[1] = _mm_mul_pd(reg_output[1], this->reg_beta_);

  // Add updated result into input registers
  reg[0] = _mm_add_pd(reg_output[0], reg[0]);
  reg[1] = _mm_add_pd(reg_output[1], reg[1]);

  // Write back in-register result into output data
  _mm_storeu_pd(out_data, reg[0]);
  _mm_storeu_pd(out_data + output_stride, reg[1]);
}


template <>
void KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_HALF>::exec(
    const FloatComplex * RESTRICT in_data, FloatComplex * RESTRICT out_data,
    const TensorIdx input_stride, const TensorIdx output_stride) const {
  // Load input data into registers
  __m128 reg_input[2];
  reg_input[0] = _mm_loadu_ps(reinterpret_cast<const float *>(in_data));
  reg_input[1] = _mm_loadu_ps(
      reinterpret_cast<const float *>(in_data + input_stride));

  // 2x2 in-register transpose
  __m128 reg[2];
  reg[0] = _mm_movelh_ps(reg_input[0], reg_input[1]);
  reg[1] = _mm_movehl_ps(reg_input[1], reg_input[0]);

  // Rescale transposed in_data
  reg[0] = _mm_mul_ps(reg[0], this->reg_alpha_);
  reg[1] = _mm_mul_ps(reg[1], this->reg_alpha_);

  // Load output data into registers
  __m128 reg_output[2];
  reg_output[0] = _mm_loadu_ps(reinterpret_cast<float *>(out_data));
  reg_output[1] = _mm_loadu_ps(
      reinterpret_cast<float *>(out_data + output_stride));

  // Update output data
  reg_output[0] = _mm_mul_ps(reg_output[0], this->reg_beta_);
  reg_output[1] = _mm_mul_ps(reg_output[1], this->reg_beta_);

  // Add updated result into input registers
  reg[0] = _mm_add_ps(reg_output[0], reg[0]);
  reg[1] = _mm_add_ps(reg_output[1], reg[1]);

  // Write back in-register result into output data
  _mm_storeu_ps(reinterpret_cast<float *>(out_data), reg[0]);
  _mm_storeu_ps(reinterpret_cast<float *>(out_data + output_stride), reg[1]);
}


template <>
void KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_HALF>::exec(
    const DoubleComplex * RESTRICT in_data, DoubleComplex * RESTRICT out_data,
    const TensorIdx, const TensorIdx) const {
  *out_data = this->reg_alpha_ * *in_data + this->reg_beta_ * *out_data;
}


template <typename FloatType>
void KernelTrans<FloatType, KernelTypeTrans::KERNEL_LINE>::exec(
    const FloatType * RESTRICT in_data, FloatType * RESTRICT out_data,
    const TensorIdx in_size, const TensorIdx out_size) const {
  using Intrin = DeducedRegType<FloatType, KernelTypeTrans::KERNEL_LINE>;
  using Deduced = DeducedFloatType<FloatType>;
  constexpr TensorUInt REG_CAP = hptc::REG_SIZE / sizeof(FloatType);

  TensorIdx idx = 0;
  for (constexpr auto step = REG_CAP * 4; idx + step <= in_size; idx += step) {
    Intrin::store(reinterpret_cast<Deduced *>(&out_data[idx]),
        Intrin::add(Intrin::mul(this->reg_alpha_,
            Intrin::load(reinterpret_cast<const Deduced *>(&in_data[idx]))),
            Intrin::mul(this->reg_beta_,
                Intrin::load(reinterpret_cast<Deduced *>(&out_data[idx])))));

    Intrin::store(reinterpret_cast<Deduced *>(&out_data[idx + REG_CAP]),
        Intrin::add(Intrin::mul(this->reg_alpha_, Intrin::load(
                reinterpret_cast<const Deduced *>(&in_data[idx + REG_CAP]))),
            Intrin::mul(this->reg_beta_, Intrin::load(
                reinterpret_cast<Deduced *>(&out_data[idx + REG_CAP])))));

    Intrin::store(reinterpret_cast<Deduced *>(&out_data[idx + REG_CAP * 2]),
        Intrin::add(Intrin::mul(this->reg_alpha_, Intrin::load(
                reinterpret_cast<const Deduced *>(&in_data[idx + REG_CAP * 2]))),
            Intrin::mul(this->reg_beta_, Intrin::load(
                reinterpret_cast<Deduced *>(&out_data[idx + REG_CAP * 2])))));

    Intrin::store(reinterpret_cast<Deduced *>(&out_data[idx + REG_CAP * 3]),
        Intrin::add(Intrin::mul(this->reg_alpha_, Intrin::load(
                reinterpret_cast<const Deduced *>(&in_data[idx + REG_CAP * 3]))),
            Intrin::mul(this->reg_beta_, Intrin::load(
                reinterpret_cast<Deduced *>(&out_data[idx + REG_CAP * 3])))));
  }

  for (constexpr auto step = REG_CAP * 2; idx + step <= in_size; idx += step) {
    Intrin::store(reinterpret_cast<Deduced *>(&out_data[idx]),
        Intrin::add(Intrin::mul(this->reg_alpha_,
            Intrin::load(reinterpret_cast<const Deduced *>(&in_data[idx]))),
            Intrin::mul(this->reg_beta_, Intrin::load(
                reinterpret_cast<Deduced *>(&out_data[idx])))));

    Intrin::store(reinterpret_cast<Deduced *>(&out_data[idx + REG_CAP]),
        Intrin::add(Intrin::mul(this->reg_alpha_, Intrin::load(
                reinterpret_cast<const Deduced *>(&in_data[idx + REG_CAP]))),
            Intrin::mul(this->reg_beta_, Intrin::load(
                reinterpret_cast<Deduced *>(&out_data[idx + REG_CAP])))));
  }

  for (; idx + REG_CAP <= in_size; idx += REG_CAP)
    Intrin::store(reinterpret_cast<Deduced *>(&out_data[idx]),
        Intrin::add(Intrin::mul(this->reg_alpha_,
            Intrin::load(reinterpret_cast<const Deduced *>(&in_data[idx]))),
            Intrin::mul(this->reg_beta_, Intrin::load(
                reinterpret_cast<Deduced *>(&out_data[idx])))));

  for (; idx < in_size; ++idx)
    out_data[idx] = this->alpha_ * in_data[idx] + this->beta_ * out_data[idx];

  if (out_size > in_size)
    std::fill(reinterpret_cast<Deduced *>(&out_data[idx]),
        reinterpret_cast<Deduced *>(&out_data[out_size]),
        static_cast<Deduced>(0.0));
}


/*
 * Explicit template instantiation definition for class KernelTrans
 */
template class KernelTrans<float, KernelTypeTrans::KERNEL_FULL>;
template class KernelTrans<double, KernelTypeTrans::KERNEL_FULL>;
template class KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_FULL>;
template class KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_FULL>;

template class KernelTrans<float, KernelTypeTrans::KERNEL_HALF>;
template class KernelTrans<double, KernelTypeTrans::KERNEL_HALF>;
template class KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_HALF>;
template class KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_HALF>;

template class KernelTrans<float, KernelTypeTrans::KERNEL_LINE>;
template class KernelTrans<double, KernelTypeTrans::KERNEL_LINE>;
template class KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_LINE>;
template class KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_LINE>;

}
