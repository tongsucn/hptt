#include <hptt/arch/ibm/kernel_trans_ibm.h>

#include <hptt/types.h>
#include <hptt/arch/compat.h>
#include <hptt/util/util_trans.h>
#include <hptt/arch/common/common_impl.h>


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
  common_trans_impl<FloatType, WIDTH, UPDATE_OUT>(data_in, data_out,
      stride_in_outld, stride_out_inld, this->alpha_, this->beta_;
}


/*
 * Implementation of linear kernel
 */
template <typename FloatType,
          bool UPDATE_OUT>
KernelTrans<FloatType, KernelTypeTrans::KERNEL_LINE, UPDATE_OUT>::KernelTrans()
    : KernelTransData<FloatType, KernelTypeTrans::KERNEL_LINE>() {
}


template <typename FloatType,
          bool UPDATE_OUT>
void KernelTrans<FloatType, KernelTypeTrans::KERNEL_LINE, UPDATE_OUT>::exec(
    const FloatType * RESTRICT data_in, FloatType * RESTRICT data_out,
    const TensorIdx size_trans, const TensorIdx) const {
  common_trans_linear_impl<FloatType, UPDATE_OUT>(data_in, data_out, size_trans,
      0, alpha_, beta_);
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
