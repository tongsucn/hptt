#include <hptc/arch/avx2/kernel_trans_avx2.h>

#include <immintrin.h>
#include <xmmintrin.h>

#include <hptc/types.h>
#include <hptc/arch/compat.h>
#include <hptc/util/util_trans.h>


namespace hptc {

/*
 * Implementation of class KernelTrans
 */
template <>
KernelTrans<float, KernelTypeTrans::KERNEL_FULL>::KernelTrans()
    : KernelTransData<float, KernelTypeTrans::KERNEL_FULL>() {
}

template <>
void KernelTrans<float, KernelTypeTrans::KERNEL_FULL>::exec(
    const float * RESTRICT data_in, float * RESTRICT data_out,
    const TensorIdx stride_in_outld, const TensorIdx stride_out_inld) const {
  using Intrin = IntrinImpl<float, KernelTypeTrans::KERNEL_FULL>;

  // Load input data into registers
  __m256 reg_input[8];
  reg_input[0] = Intrin::load(data_in);
  reg_input[1] = Intrin::load(data_in + stride_in_outld);
  reg_input[2] = Intrin::load(data_in + 2 * stride_in_outld);
  reg_input[3] = Intrin::load(data_in + 3 * stride_in_outld);
  reg_input[4] = Intrin::load(data_in + 4 * stride_in_outld);
  reg_input[5] = Intrin::load(data_in + 5 * stride_in_outld);
  reg_input[6] = Intrin::load(data_in + 6 * stride_in_outld);
  reg_input[7] = Intrin::load(data_in + 7 * stride_in_outld);

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
  reg_input[0] = Intrin::mul(reg_input[0], this->reg_alpha_);
  reg_input[1] = Intrin::mul(reg_input[1], this->reg_alpha_);
  reg_input[2] = Intrin::mul(reg_input[2], this->reg_alpha_);
  reg_input[3] = Intrin::mul(reg_input[3], this->reg_alpha_);
  reg_input[4] = Intrin::mul(reg_input[4], this->reg_alpha_);
  reg_input[5] = Intrin::mul(reg_input[5], this->reg_alpha_);
  reg_input[6] = Intrin::mul(reg_input[6], this->reg_alpha_);
  reg_input[7] = Intrin::mul(reg_input[7], this->reg_alpha_);

  // Load output data into registers
  __m256 reg_output[8];
  reg_output[0] = Intrin::load(data_out);
  reg_output[1] = Intrin::load(data_out + stride_out_inld);
  reg_output[2] = Intrin::load(data_out + 2 * stride_out_inld);
  reg_output[3] = Intrin::load(data_out + 3 * stride_out_inld);
  reg_output[4] = Intrin::load(data_out + 4 * stride_out_inld);
  reg_output[5] = Intrin::load(data_out + 5 * stride_out_inld);
  reg_output[6] = Intrin::load(data_out + 6 * stride_out_inld);
  reg_output[7] = Intrin::load(data_out + 7 * stride_out_inld);

  // Update output data
  reg_output[0] = Intrin::mul(reg_output[0], this->reg_beta_);
  reg_output[1] = Intrin::mul(reg_output[1], this->reg_beta_);
  reg_output[2] = Intrin::mul(reg_output[2], this->reg_beta_);
  reg_output[3] = Intrin::mul(reg_output[3], this->reg_beta_);
  reg_output[4] = Intrin::mul(reg_output[4], this->reg_beta_);
  reg_output[5] = Intrin::mul(reg_output[5], this->reg_beta_);
  reg_output[6] = Intrin::mul(reg_output[6], this->reg_beta_);
  reg_output[7] = Intrin::mul(reg_output[7], this->reg_beta_);

  // Add updated result into input registers
  reg_input[0] = Intrin::add(reg_output[0], reg_input[0]);
  reg_input[1] = Intrin::add(reg_output[1], reg_input[1]);
  reg_input[2] = Intrin::add(reg_output[2], reg_input[2]);
  reg_input[3] = Intrin::add(reg_output[3], reg_input[3]);
  reg_input[4] = Intrin::add(reg_output[4], reg_input[4]);
  reg_input[5] = Intrin::add(reg_output[5], reg_input[5]);
  reg_input[6] = Intrin::add(reg_output[6], reg_input[6]);
  reg_input[7] = Intrin::add(reg_output[7], reg_input[7]);

  // Write back in-register result into output data
  Intrin::store(data_out, reg_input[0]);
  Intrin::store(data_out + stride_out_inld, reg_input[1]);
  Intrin::store(data_out + 2 * stride_out_inld, reg_input[2]);
  Intrin::store(data_out + 3 * stride_out_inld, reg_input[3]);
  Intrin::store(data_out + 4 * stride_out_inld, reg_input[4]);
  Intrin::store(data_out + 5 * stride_out_inld, reg_input[5]);
  Intrin::store(data_out + 6 * stride_out_inld, reg_input[6]);
  Intrin::store(data_out + 7 * stride_out_inld, reg_input[7]);
}


template <>
KernelTrans<double, KernelTypeTrans::KERNEL_FULL>::KernelTrans()
    : KernelTransData<double, KernelTypeTrans::KERNEL_FULL>() {
}

template <>
void KernelTrans<double, KernelTypeTrans::KERNEL_FULL>::exec(
    const double * RESTRICT data_in, double * RESTRICT data_out,
    const TensorIdx stride_in_outld, const TensorIdx stride_out_inld) const {
  using Intrin = IntrinImpl<double, KernelTypeTrans::KERNEL_FULL>;

  // Load input data into registers
  __m256d reg_input[4];
  reg_input[0] = Intrin::load(data_in);
  reg_input[1] = Intrin::load(data_in + stride_in_outld);
  reg_input[2] = Intrin::load(data_in + 2 * stride_in_outld);
  reg_input[3] = Intrin::load(data_in + 3 * stride_in_outld);

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
  reg_input[0] = Intrin::mul(reg_input[0], this->reg_alpha_);
  reg_input[1] = Intrin::mul(reg_input[1], this->reg_alpha_);
  reg_input[2] = Intrin::mul(reg_input[2], this->reg_alpha_);
  reg_input[3] = Intrin::mul(reg_input[3], this->reg_alpha_);

  // Load output data into registers
  __m256d reg_output[4];
  reg_output[0] = Intrin::load(data_out);
  reg_output[1] = Intrin::load(data_out + stride_out_inld);
  reg_output[2] = Intrin::load(data_out + 2 * stride_out_inld);
  reg_output[3] = Intrin::load(data_out + 3 * stride_out_inld);

  // Update output data
  reg_output[0] = Intrin::mul(reg_output[0], this->reg_beta_);
  reg_output[1] = Intrin::mul(reg_output[1], this->reg_beta_);
  reg_output[2] = Intrin::mul(reg_output[2], this->reg_beta_);
  reg_output[3] = Intrin::mul(reg_output[3], this->reg_beta_);

  // Add updated result into input registers
  reg_input[0] = Intrin::add(reg_output[0], reg_input[0]);
  reg_input[1] = Intrin::add(reg_output[1], reg_input[1]);
  reg_input[2] = Intrin::add(reg_output[2], reg_input[2]);
  reg_input[3] = Intrin::add(reg_output[3], reg_input[3]);

  // Write back in-register result into output data
  Intrin::store(data_out, reg_input[0]);
  Intrin::store(data_out + stride_out_inld, reg_input[1]);
  Intrin::store(data_out + 2 * stride_out_inld, reg_input[2]);
  Intrin::store(data_out + 3 * stride_out_inld, reg_input[3]);
}


template <>
KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_FULL>::KernelTrans()
    : KernelTransData<FloatComplex, KernelTypeTrans::KERNEL_FULL>() {
}

template <>
void KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_FULL>::exec(
    const FloatComplex * RESTRICT data_in, FloatComplex * RESTRICT data_out,
    const TensorIdx stride_in_outld, const TensorIdx stride_out_inld) const {
  using Intrin = IntrinImpl<FloatComplex, KernelTypeTrans::KERNEL_FULL>;

  // Load input data into registers
  __m256 reg_input[4];
  reg_input[0] = Intrin::load(data_in);
  reg_input[1] = Intrin::load(data_in + stride_in_outld);
  reg_input[2] = Intrin::load(data_in + 2 * stride_in_outld);
  reg_input[3] = Intrin::load(data_in + 3 * stride_in_outld);

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
  reg_input[0] = Intrin::mul(reg_input[0], this->reg_alpha_);
  reg_input[1] = Intrin::mul(reg_input[1], this->reg_alpha_);
  reg_input[2] = Intrin::mul(reg_input[2], this->reg_alpha_);
  reg_input[3] = Intrin::mul(reg_input[3], this->reg_alpha_);

  // Load output data into registers
  __m256 reg_output[4];
  reg_output[0] = Intrin::load(data_out);
  reg_output[1] = Intrin::load(data_out + stride_out_inld);
  reg_output[2] = Intrin::load(data_out + 2 * stride_out_inld);
  reg_output[3] = Intrin::load(data_out + 3 * stride_out_inld);

  // Update output data
  reg_output[0] = Intrin::mul(reg_output[0], this->reg_beta_);
  reg_output[1] = Intrin::mul(reg_output[1], this->reg_beta_);
  reg_output[2] = Intrin::mul(reg_output[2], this->reg_beta_);
  reg_output[3] = Intrin::mul(reg_output[3], this->reg_beta_);

  // Add updated result into input registers
  reg_input[0] = Intrin::add(reg_output[0], reg_input[0]);
  reg_input[1] = Intrin::add(reg_output[1], reg_input[1]);
  reg_input[2] = Intrin::add(reg_output[2], reg_input[2]);
  reg_input[3] = Intrin::add(reg_output[3], reg_input[3]);

  // Write back in-register result into output data
  Intrin::store(data_out, reg_input[0]);
  Intrin::store(data_out + stride_out_inld, reg_input[1]);
  Intrin::store(data_out + 2 * stride_out_inld, reg_input[2]);
  Intrin::store(data_out + 3 * stride_out_inld, reg_input[3]);
}


template <>
KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_FULL>::KernelTrans()
    : KernelTransData<DoubleComplex, KernelTypeTrans::KERNEL_FULL>() {
}

template <>
void KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_FULL>::exec(
    const DoubleComplex * RESTRICT data_in, DoubleComplex * RESTRICT data_out,
    const TensorIdx stride_in_outld, const TensorIdx stride_out_inld) const {
  using Intrin = IntrinImpl<DoubleComplex, KernelTypeTrans::KERNEL_FULL>;

  // Load input data into registers
  __m256d reg_input[2];
  reg_input[0] = Intrin::load(data_in);
  reg_input[1] = Intrin::load(data_in + stride_in_outld);

  // 2x2 in-register transpose
  __m256d reg[2];
  reg[0] = _mm256_permute2f128_pd(reg_input[1], reg_input[0], 0x2);
  reg[1] = _mm256_permute2f128_pd(reg_input[1], reg_input[0], 0x13);
  reg_input[0] = reg[0];
  reg_input[1] = reg[1];

  // Rescale transposed input data
  reg_input[0] = Intrin::mul(reg_input[0], this->reg_alpha_);
  reg_input[1] = Intrin::mul(reg_input[1], this->reg_alpha_);

  // Load output data into registers
  __m256d reg_output[2];
  reg_output[0] = Intrin::load(data_out);
  reg_output[1] = Intrin::load(data_out + stride_out_inld);

  // Update output data
  reg_output[0] = Intrin::mul(reg_output[0], this->reg_beta_);
  reg_output[1] = Intrin::mul(reg_output[1], this->reg_beta_);

  // Add updated result into input registers
  reg_input[0] = Intrin::add(reg_output[0], reg_input[0]);
  reg_input[1] = Intrin::add(reg_output[1], reg_input[1]);

  // Write back in-register result into output data
  Intrin::store(data_out, reg_input[0]);
  Intrin::store(data_out + stride_out_inld, reg_input[1]);
}


template <>
KernelTrans<float, KernelTypeTrans::KERNEL_HALF>::KernelTrans()
    : KernelTransData<float, KernelTypeTrans::KERNEL_HALF>() {
}

template <>
void KernelTrans<float, KernelTypeTrans::KERNEL_HALF>::exec(
    const float * RESTRICT data_in, float * RESTRICT data_out,
    const TensorIdx stride_in_outld, const TensorIdx stride_out_inld) const {
  using Intrin = IntrinImpl<float, KernelTypeTrans::KERNEL_HALF>;

  // Load input data into registers
  __m128 reg_input[4];
  reg_input[0] = Intrin::load(data_in);
  reg_input[1] = Intrin::load(data_in + stride_in_outld);
  reg_input[2] = Intrin::load(data_in + 2 * stride_in_outld);
  reg_input[3] = Intrin::load(data_in + 3 * stride_in_outld);

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

  // Rescale transposed data_in
  reg_input[0] = Intrin::mul(reg_input[0], this->reg_alpha_);
  reg_input[1] = Intrin::mul(reg_input[1], this->reg_alpha_);
  reg_input[2] = Intrin::mul(reg_input[2], this->reg_alpha_);
  reg_input[3] = Intrin::mul(reg_input[3], this->reg_alpha_);

  // Load output data into registers
  __m128 reg_output[4];
  reg_output[0] = Intrin::load(data_out);
  reg_output[1] = Intrin::load(data_out + stride_out_inld);
  reg_output[2] = Intrin::load(data_out + 2 * stride_out_inld);
  reg_output[3] = Intrin::load(data_out + 3 * stride_out_inld);

  // Update output data
  reg_output[0] = Intrin::mul(reg_output[0], this->reg_beta_);
  reg_output[1] = Intrin::mul(reg_output[1], this->reg_beta_);
  reg_output[2] = Intrin::mul(reg_output[2], this->reg_beta_);
  reg_output[3] = Intrin::mul(reg_output[3], this->reg_beta_);

  // Add updated result into input registers
  reg_input[0] = Intrin::add(reg_output[0], reg_input[0]);
  reg_input[1] = Intrin::add(reg_output[1], reg_input[1]);
  reg_input[2] = Intrin::add(reg_output[2], reg_input[2]);
  reg_input[3] = Intrin::add(reg_output[3], reg_input[3]);

  // Write back in-register result into output data
  Intrin::store(data_out, reg_input[0]);
  Intrin::store(data_out + stride_out_inld, reg_input[1]);
  Intrin::store(data_out + 2 * stride_out_inld, reg_input[2]);
  Intrin::store(data_out + 3 * stride_out_inld, reg_input[3]);
}


template <>
KernelTrans<double, KernelTypeTrans::KERNEL_HALF>::KernelTrans()
    : KernelTransData<double, KernelTypeTrans::KERNEL_HALF>() {
}

template <>
void KernelTrans<double, KernelTypeTrans::KERNEL_HALF>::exec(
    const double * RESTRICT data_in, double * RESTRICT data_out,
    const TensorIdx stride_in_outld, const TensorIdx stride_out_inld) const {
  using Intrin = IntrinImpl<double, KernelTypeTrans::KERNEL_HALF>;

  // Load input data into registers
  __m128d reg_input[2];
  reg_input[0] = Intrin::load(data_in);
  reg_input[1] = Intrin::load(data_in + stride_in_outld);

  // 2x2 in-register transpose
  __m128d reg[2];
  reg[0] = _mm_unpacklo_pd(reg_input[0], reg_input[1]);
  reg[1] = _mm_unpackhi_pd(reg_input[0], reg_input[1]);

  // Rescale transposed data_in
  reg[0] = Intrin::mul(reg[0], this->reg_alpha_);
  reg[1] = Intrin::mul(reg[1], this->reg_alpha_);

  // Load output data into registers
  __m128d reg_output[2];
  reg_output[0] = Intrin::load(data_out);
  reg_output[1] = Intrin::load(data_out + stride_out_inld);

  // Update output data
  reg_output[0] = Intrin::mul(reg_output[0], this->reg_beta_);
  reg_output[1] = Intrin::mul(reg_output[1], this->reg_beta_);

  // Add updated result into input registers
  reg[0] = Intrin::add(reg_output[0], reg[0]);
  reg[1] = Intrin::add(reg_output[1], reg[1]);

  // Write back in-register result into output data
  Intrin::store(data_out, reg[0]);
  Intrin::store(data_out + stride_out_inld, reg[1]);
}


template <>
KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_HALF>::KernelTrans()
    : KernelTransData<FloatComplex, KernelTypeTrans::KERNEL_HALF>() {
}

template <>
void KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_HALF>::exec(
    const FloatComplex * RESTRICT data_in, FloatComplex * RESTRICT data_out,
    const TensorIdx stride_in_outld, const TensorIdx stride_out_inld) const {
  using Intrin = IntrinImpl<FloatComplex, KernelTypeTrans::KERNEL_HALF>;

  // Load input data into registers
  __m128 reg_input[2];
  reg_input[0] = Intrin::load(data_in);
  reg_input[1] = Intrin::load(data_in + stride_in_outld);

  // 2x2 in-register transpose
  __m128 reg[2];
  reg[0] = _mm_movelh_ps(reg_input[0], reg_input[1]);
  reg[1] = _mm_movehl_ps(reg_input[1], reg_input[0]);

  // Rescale transposed data_in
  reg[0] = Intrin::mul(reg[0], this->reg_alpha_);
  reg[1] = Intrin::mul(reg[1], this->reg_alpha_);

  // Load output data into registers
  __m128 reg_output[2];
  reg_output[0] = Intrin::load(data_out);
  reg_output[1] = Intrin::load(data_out + stride_out_inld);

  // Update output data
  reg_output[0] = Intrin::mul(reg_output[0], this->reg_beta_);
  reg_output[1] = Intrin::mul(reg_output[1], this->reg_beta_);

  // Add updated result into input registers
  reg[0] = Intrin::add(reg_output[0], reg[0]);
  reg[1] = Intrin::add(reg_output[1], reg[1]);

  // Write back in-register result into output data
  Intrin::store(data_out, reg[0]);
  Intrin::store(data_out + stride_out_inld, reg[1]);
}


template <>
KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_HALF>::KernelTrans()
    : KernelTransData<DoubleComplex, KernelTypeTrans::KERNEL_HALF>() {
}

template <>
void KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_HALF>::exec(
    const DoubleComplex * RESTRICT data_in, DoubleComplex * RESTRICT data_out,
    const TensorIdx, const TensorIdx) const {
  *data_out = this->reg_alpha_ * *data_in + this->reg_beta_ * *data_out;
}


template <typename FloatType>
KernelTrans<FloatType, KernelTypeTrans::KERNEL_LINE>::KernelTrans()
    : KernelTransData<FloatType, KernelTypeTrans::KERNEL_LINE>(),
      stride_in_inld_(1), stride_in_outld_(1), stride_out_inld_(1),
      stride_out_outld_(1), size_kn_inld_(1), size_kn_outld_(1) {
}


template <typename FloatType>
void KernelTrans<FloatType, KernelTypeTrans::KERNEL_LINE>::set_wrapper_loop(
    const TensorIdx stride_in_inld, const TensorIdx stride_in_outld,
    const TensorIdx stride_out_inld, const TensorIdx stride_out_outld,
    const TensorUInt size_kn_inld, const TensorUInt size_kn_outld) {
  this->stride_in_inld_ = stride_in_inld;
  this->stride_in_outld_ = stride_in_outld;
  this->stride_out_inld_ = stride_out_inld;
  this->stride_out_outld_ = stride_out_outld;
  this->size_kn_inld_ = size_kn_inld > 0 ? size_kn_inld : 1;
  this->size_kn_outld_ = size_kn_outld > 0 ? size_kn_outld : 1;
}


template <typename FloatType>
void KernelTrans<FloatType, KernelTypeTrans::KERNEL_LINE>::exec(
    const FloatType * RESTRICT data_in, FloatType * RESTRICT data_out,
    const TensorIdx size_trans, const TensorIdx size_pad) const {
  using Intrin = IntrinImpl<FloatType, KernelTypeTrans::KERNEL_LINE>;
  constexpr TensorUInt REG_CAP = hptc::SIZE_REG / sizeof(FloatType);

  for (TensorUInt out_idx = 0; out_idx < this->size_kn_outld_; ++out_idx) {
    for (TensorUInt in_idx = 0; in_idx < this->size_kn_inld_; ++in_idx) {
      auto ptr_in = data_in + this->stride_in_inld_ * in_idx
          + this->stride_in_outld_ * out_idx;
      auto ptr_out = data_out + this->stride_out_inld_ * in_idx
          + this->stride_out_outld_ * out_idx;

      TensorIdx idx = 0;
      for (constexpr auto step = REG_CAP * 2; idx + step <= size_trans;
          idx += step, ptr_in += step, ptr_out += step) {
        Intrin::store(ptr_out, Intrin::add(
            Intrin::mul(this->reg_alpha_, Intrin::load(ptr_in)),
            Intrin::mul(this->reg_beta_, Intrin::load(ptr_out))));

        Intrin::store(ptr_out + REG_CAP, Intrin::add(
            Intrin::mul(this->reg_alpha_, Intrin::load(ptr_in + REG_CAP)),
            Intrin::mul(this->reg_beta_, Intrin::load(ptr_out + REG_CAP))));
      }

      for (constexpr auto step = REG_CAP; idx + step <= size_trans;
          idx += step, ptr_in += step, ptr_out += step)
        Intrin::store(ptr_out, Intrin::add(
            Intrin::mul(this->reg_alpha_, Intrin::load(ptr_in)),
            Intrin::mul(this->reg_beta_, Intrin::load(ptr_out))));

      for (; idx < size_trans; ++idx)
        ptr_out[idx] = this->alpha_ * ptr_in[idx] + this->beta_ * ptr_out[idx];
    }
  }
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
