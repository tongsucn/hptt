#include <hptt/arch/ibm/kernel_trans_ibm.h>

#include <immintrin.h>
#include <xmmintrin.h>

#include <hptt/types.h>
#include <hptt/arch/compat.h>
#include <hptt/util/util_trans.h>


namespace hptt {

/*
 * Implementation of class KernelTrans and its (partial) specializations
 */
template <bool UPDATE_OUT>
KernelTrans<double, KernelTypeTrans::KERNEL_FULL, UPDATE_OUT>::KernelTrans()
    : KernelTransData<double, KernelTypeTrans::KERNEL_FULL>() {
}

template <bool UPDATE_OUT>
void KernelTrans<double, KernelTypeTrans::KERNEL_FULL, UPDATE_OUT>::exec(
    const double * RESTRICT data_in, double * RESTRICT data_out,
    const TensorIdx stride_in_outld, const TensorIdx stride_out_inld) const {
  using Intrin = IntrinImpl<double, KernelTypeTrans::KERNEL_FULL>;
  using Reg = RegType<double, KernelTypeTrans::KERNEL_FULL>;

  // Load output data into registers
  Reg reg_in[4];
  reg_in[0] = Intrin::load(data_in);
  reg_in[1] = Intrin::load(data_in + stride_in_outld);
  reg_in[2] = Intrin::load(data_in + 2 * stride_in_outld);
  reg_in[3] = Intrin::load(data_in + 3 * stride_in_outld);

  // 4x4 in-register transpose
  Reg mid_res[4];
  mid_res[0] = vec_perm(reg_in[0], reg_in[1], Intrin::rule[0]); // 0, 4, 2, 6
  mid_res[1] = vec_perm(reg_in[0], reg_in[1], Intrin::rule[1]); // 1, 5, 3, 7
  mid_res[2] = vec_perm(reg_in[2], reg_in[3], Intrin::rule[0]); // 8, 12, 10, 14
  mid_res[3] = vec_perm(reg_in[2], reg_in[3], Intrin::rule[1]); // 9, 13, 11, 15


  reg_in[0] = vec_perm(mid_res[0], mid_res[2], Intrin::rule[2]); // 0, 4, 8, 12
  reg_in[1] = vec_perm(mid_res[1], mid_res[3], Intrin::rule[2]); // 1, 5, 9, 13
  reg_in[2] = vec_perm(mid_res[0], mid_res[2], Intrin::rule[3]); // 2, 6, 10, 14
  reg_in[3] = vec_perm(mid_res[1], mid_res[3], Intrin::rule[4]); // 3, 7, 11, 15

  // Rescaled transposed input data
  reg_in[0] = Intrin::mul(reg_in[0], this->reg_alpha_);
  reg_in[1] = Intrin::mul(reg_in[1], this->reg_alpha_);
  reg_in[2] = Intrin::mul(reg_in[2], this->reg_alpha_);
  reg_in[3] = Intrin::mul(reg_in[3], this->reg_alpha_);

  if (UPDATE_OUT) {
    // Load output data into registers
    Reg reg_out[4];
    reg_out[0] = Introut::load(data_out);
    reg_out[1] = Introut::load(data_out + stride_out_inld);
    reg_out[2] = Introut::load(data_out + 2 * stride_out_inld);
    reg_out[3] = Introut::load(data_out + 3 * stride_out_inld);

    reg_out[0] = Intrin::madd(reg_out[0], this->reg_beta_, reg_in[0]);
    reg_out[1] = Intrin::madd(reg_out[1], this->reg_beta_, reg_in[1]);
    reg_out[2] = Intrin::madd(reg_out[2], this->reg_beta_, reg_in[2]);
    reg_out[3] = Intrin::madd(reg_out[3], this->reg_beta_, reg_in[3]);
  }

  // Write back in-register result into output data
  Intrin::store(data_out, reg_out[0]);
  Intrin::store(data_out + stride_out_inld, reg_out[1]);
  Intrin::store(data_out + 2 * stride_out_inld, reg_out[2]);
  Intrin::store(data_out + 3 * stride_out_inld, reg_out[3]);
}


template <typename FloatType,
          KernelTypeTrans TYPE,
          bool UPDATE_OUT>
KernelTrans<FloatType, TYPE, UPDATE_OUT>::KernelTrans()
    : KernelTransData<FloatType, TYPE>() {
}

template <typename FloatType,
          KernelTypeTrans TYPE,
          bool UPDATE_OUT>
void KernelTrans<FloatType, TYPE, UPDATE_OUT>::exec(
    const FloatType * RESTRICT data_in, FloatType * RESTRICT data_out,
    const TensorIdx stride_in_outld, const TensorIdx stride_out_inld) const {
  constexpr auto WIDTH = KernelTrans<FloatType, TYPE>::KN_WIDTH;

  if (UPDATE_OUT) {
#pragma omp simd collapse(2)
    for (TensorUInt idx_row = 0; idx_row < WIDTH; ++idx_row) {
      for (TensorUInt idx_col = 0; idx_col < WIDTH; ++idx_col) {
        const TensorIdx offset_in = idx_col + idx_row * stride_out_inld;
        const TensorIdx offset_out = idx_row + idx_col * stride_out_inld;
        data_out[offset_out] = this->reg_alpha_ * data_in[offset_in]
            + this->reg_beta_ * data_out[offset_out];
      }
    }
  }
  else {
#pragma omp simd collapse(2)
    for (TensorUInt idx_row = 0; idx_row < WIDTH; ++idx_row) {
      for (TensorUInt idx_col = 0; idx_col < WIDTH; ++idx_col) {
        const TensorIdx offset_in = idx_col + idx_row * stride_out_inld;
        const TensorIdx offset_out = idx_row + idx_col * stride_out_inld;
        data_out[offset_out] = this->reg_alpha_ * data_in[offset_in];
      }
    }
  }
}


/*
 * Implementation of linear kernel
 */
template <typename FloatType,
          bool UPDATE_OUT>
KernelTrans<FloatType, KernelTypeTrans::KERNEL_LINE, UPDATE_OUT>::KernelTrans()
    : KernelTransData<FloatType, KernelTypeTrans::KERNEL_LINE>(),
      stride_in_inld_(1), stride_in_outld_(1), stride_out_inld_(1),
      stride_out_outld_(1), size_kn_inld_(1), size_kn_outld_(1) {
}


template <typename FloatType,
          bool UPDATE_OUT>
void KernelTrans<FloatType, KernelTypeTrans::KERNEL_LINE, UPDATE_OUT>::
set_wrapper_loop(const TensorIdx stride_in_inld,
    const TensorIdx stride_in_outld, const TensorIdx stride_out_inld,
    const TensorIdx stride_out_outld, const TensorUInt size_kn_inld,
    const TensorUInt size_kn_outld) {
  this->stride_in_inld_ = stride_in_inld;
  this->stride_in_outld_ = stride_in_outld;
  this->stride_out_inld_ = stride_out_inld;
  this->stride_out_outld_ = stride_out_outld;
  this->size_kn_inld_ = size_kn_inld > 0 ? size_kn_inld : 1;
  this->size_kn_outld_ = size_kn_outld > 0 ? size_kn_outld : 1;
}


template <typename FloatType,
          bool UPDATE_OUT>
void KernelTrans<FloatType, KernelTypeTrans::KERNEL_LINE, UPDATE_OUT>::exec(
    const FloatType * RESTRICT data_in, FloatType * RESTRICT data_out,
    const TensorIdx size_trans, const TensorIdx size_pad) const {
  if (UPDATE_OUT) {
    for (TensorUInt out_idx = 0; out_idx < this->size_kn_outld_; ++out_idx) {
      for (TensorUInt in_idx = 0; in_idx < this->size_kn_inld_; ++in_idx) {
        const FloatType * RESTRICT ptr_in
            = data_in + this->stride_in_inld_ * in_idx
            + this->stride_in_outld_ * out_idx;
        FloatType * RESTRICT ptr_out
            = data_out + this->stride_out_inld_ * in_idx
            + this->stride_out_outld_ * out_idx;

#pragma omp simd
        for (TensorIdx idx = 0; idx < size_trans; ++idx)
          out_ptr[idx] = this->alpha_ * in_ptr[idx]
              + this->beta_ * out_ptr[idx];
      }
    }
  }
  else {
    for (TensorUInt out_idx = 0; out_idx < this->size_kn_outld_; ++out_idx) {
      for (TensorUInt in_idx = 0; in_idx < this->size_kn_inld_; ++in_idx) {
        const FloatType * RESTRICT ptr_in
            = data_in + this->stride_in_inld_ * in_idx
            + this->stride_in_outld_ * out_idx;
        FloatType * RESTRICT ptr_out
            = data_out + this->stride_out_inld_ * in_idx
            + this->stride_out_outld_ * out_idx;

#pragma omp simd
        for (TensorIdx idx = 0; idx < size_trans; ++idx)
          out_ptr[idx] = this->alpha_ * in_ptr[idx];
      }
    }
  }
}


/*
 * Explicit template instantiation definition for class KernelTrans
 */
template class KernelTrans<float, KernelTypeTrans::KERNEL_FULL, true>;
template class KernelTrans<double, KernelTypeTrans::KERNEL_FULL, true>;
template class KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_FULL, true>;
template class KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_FULL, true>;

template class KernelTrans<float, KernelTypeTrans::KERNEL_HALF, true>;
template class KernelTrans<double, KernelTypeTrans::KERNEL_HALF, true>;
template class KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_HALF, true>;
template class KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_HALF, true>;

template class KernelTrans<float, KernelTypeTrans::KERNEL_LINE, true>;
template class KernelTrans<double, KernelTypeTrans::KERNEL_LINE, true>;
template class KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_LINE, true>;
template class KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_LINE, true>;

template class KernelTrans<float, KernelTypeTrans::KERNEL_FULL, false>;
template class KernelTrans<double, KernelTypeTrans::KERNEL_FULL, false>;
template class KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_FULL, false>;
template class KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_FULL, false>;

template class KernelTrans<float, KernelTypeTrans::KERNEL_HALF, false>;
template class KernelTrans<double, KernelTypeTrans::KERNEL_HALF, false>;
template class KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_HALF, false>;
template class KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_HALF, false>;

template class KernelTrans<float, KernelTypeTrans::KERNEL_LINE, false>;
template class KernelTrans<double, KernelTypeTrans::KERNEL_LINE, false>;
template class KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_LINE, false>;
template class KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_LINE, false>;

}
