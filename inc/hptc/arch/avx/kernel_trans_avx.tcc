#pragma once
#ifndef HPTC_ARCH_AVX_KERNEL_TRANS_AVX_TCC_
#define HPTC_ARCH_AVX_KERNEL_TRANS_AVX_TCC_

/*
 * Intrinsics wrappers
 */
template <typename FloatType,
          KernelTypeTrans TYPE,
          typename Selected = void>
struct IntrinImpl {
};

template <typename FloatType,
          KernelTypeTrans TYPE>
struct IntrinImpl<FloatType, TYPE,
    Enable<TypeSelector<FloatType, TYPE>::fl_sc>> {
  using Deduced = DeducedFloatType<FloatType>;
  using Reg = RegType<FloatType, TYPE>;

  static HPTC_INL Reg set_reg(const Deduced coef) {
    return _mm256_set1_ps(coef);
  }
  static HPTC_INL Reg load(const FloatType * RESTRICT target) {
    return _mm256_loadu_ps(reinterpret_cast<const Deduced *>(target));
  }
  static HPTC_INL void store(FloatType * RESTRICT target, const Reg &reg) {
    _mm256_storeu_ps(reinterpret_cast<Deduced *>(target), reg);
  }
  static HPTC_INL Reg add(const Reg &reg_a, const Reg &reg_b) {
    return _mm256_add_ps(reg_a, reg_b);
  }
  static HPTC_INL Reg mul(const Reg &reg_a, const Reg &reg_b) {
    return _mm256_mul_ps(reg_a, reg_b);
  }
};

template <typename FloatType,
          KernelTypeTrans TYPE>
struct IntrinImpl<FloatType, TYPE,
    Enable<TypeSelector<FloatType, TYPE>::fl_dz>> {
  using Deduced = DeducedFloatType<FloatType>;
  using Reg = RegType<FloatType, TYPE>;

  static HPTC_INL Reg set_reg(const Deduced coef) {
    return _mm256_set1_pd(coef);
  }
  static HPTC_INL Reg load(const FloatType * RESTRICT target) {
    return _mm256_loadu_pd(reinterpret_cast<const Deduced *>(target));
  }
  static HPTC_INL void store(FloatType * RESTRICT target, const Reg &reg) {
    _mm256_storeu_pd(reinterpret_cast<Deduced *>(target), reg);
  }
  static HPTC_INL Reg add(const Reg &reg_a, const Reg &reg_b) {
    return _mm256_add_pd(reg_a, reg_b);
  }
  static HPTC_INL Reg mul(const Reg &reg_a, const Reg &reg_b) {
    return _mm256_mul_pd(reg_a, reg_b);
  }
};

template <typename FloatType,
          KernelTypeTrans TYPE>
struct IntrinImpl<FloatType, TYPE,
    Enable<TypeSelector<FloatType, TYPE>::h_sc>> {
  using Deduced = DeducedFloatType<FloatType>;
  using Reg = RegType<FloatType, TYPE>;

  static HPTC_INL RegType<FloatType, TYPE> set_reg(const Deduced coef) {
    return _mm_set1_ps(coef);
  }
  static HPTC_INL Reg load(const FloatType * RESTRICT target) {
    return _mm_loadu_ps(reinterpret_cast<const Deduced *>(target));
  }
  static HPTC_INL void store(FloatType * RESTRICT target, const Reg &reg) {
    _mm_storeu_ps(reinterpret_cast<Deduced *>(target), reg);
  }
  static HPTC_INL Reg add(const Reg &reg_a, const Reg &reg_b) {
    return _mm_add_ps(reg_a, reg_b);
  }
  static HPTC_INL Reg mul(const Reg &reg_a, const Reg &reg_b) {
    return _mm_mul_ps(reg_a, reg_b);
  }
};

template <typename FloatType,
          KernelTypeTrans TYPE>
struct IntrinImpl<FloatType, TYPE, Enable<TypeSelector<FloatType, TYPE>::h_d>> {
  using Deduced = DeducedFloatType<FloatType>;
  using Reg = RegType<FloatType, TYPE>;

  static HPTC_INL RegType<FloatType, TYPE> set_reg(const Deduced coef) {
    return _mm_set1_pd(coef);
  }
  static HPTC_INL Reg load(const FloatType * RESTRICT target) {
    return _mm_loadu_pd(reinterpret_cast<const Deduced *>(target));
  }
  static HPTC_INL void store(FloatType * RESTRICT target, const Reg &reg) {
    _mm_storeu_pd(reinterpret_cast<Deduced *>(target), reg);
  }
  static HPTC_INL Reg add(const Reg &reg_a, const Reg &reg_b) {
    return _mm_add_pd(reg_a, reg_b);
  }
  static HPTC_INL Reg mul(const Reg &reg_a, const Reg &reg_b) {
    return _mm_mul_pd(reg_a, reg_b);
  }
};

template <typename FloatType,
          KernelTypeTrans TYPE>
struct IntrinImpl<FloatType, TYPE, Enable<TypeSelector<FloatType, TYPE>::h_z>> {
  using Deduced = DeducedFloatType<FloatType>;
  using Reg = RegType<FloatType, TYPE>;

  static HPTC_INL RegType<FloatType, TYPE> set_reg(const Deduced coef) {
    return coef;
  }
};


/*
 * Implementation of class KernelTransData
 */
template <typename FloatType,
          KernelTypeTrans TYPE>
KernelTransData<FloatType, TYPE>::KernelTransData()
    : reg_alpha_(), reg_beta_(), alpha_(), beta_() {
}


template <typename FloatType,
          KernelTypeTrans TYPE>
void KernelTransData<FloatType, TYPE>::set_coef(
    const DeducedFloatType<FloatType> alpha,
    const DeducedFloatType<FloatType> beta) {
  this->alpha_ = alpha, this->beta_ = beta;
  this->reg_alpha_ = IntrinImpl<FloatType, TYPE>::set_reg(this->alpha_);
  this->reg_beta_ = IntrinImpl<FloatType, TYPE>::set_reg(this->beta_);
}


/*
 * Specializations of class KernelTrans
 */
template <>
KernelTrans<float, KernelTypeTrans::KERNEL_FULL>::KernelTrans();
template <>
void KernelTrans<float, KernelTypeTrans::KERNEL_FULL>::exec(
    const float * RESTRICT data_in, float * RESTRICT data_out,
    const TensorIdx stride_in_outld, const TensorIdx stride_out_inld) const;

template <>
KernelTrans<double, KernelTypeTrans::KERNEL_FULL>::KernelTrans();
template <>
void KernelTrans<double, KernelTypeTrans::KERNEL_FULL>::exec(
    const double * RESTRICT data_in, double * RESTRICT data_out,
    const TensorIdx stride_in_outld, const TensorIdx stride_out_inld) const;

template <>
KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_FULL>::KernelTrans();
template <>
void KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_FULL>::exec(
    const FloatComplex * RESTRICT data_in, FloatComplex * RESTRICT data_out,
    const TensorIdx stride_in_outld, const TensorIdx stride_out_inld) const;

template <>
KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_FULL>::KernelTrans();
template <>
void KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_FULL>::exec(
    const DoubleComplex * RESTRICT data_in, DoubleComplex * RESTRICT data_out,
    const TensorIdx stride_in_outld, const TensorIdx stride_out_inld) const;


template <>
KernelTrans<float, KernelTypeTrans::KERNEL_HALF>::KernelTrans();
template <>
void KernelTrans<float, KernelTypeTrans::KERNEL_HALF>::exec(
    const float * RESTRICT data_in, float * RESTRICT data_out,
    const TensorIdx stride_in_outld, const TensorIdx stride_out_inld) const;

template <>
KernelTrans<double, KernelTypeTrans::KERNEL_HALF>::KernelTrans();
template <>
void KernelTrans<double, KernelTypeTrans::KERNEL_HALF>::exec(
    const double * RESTRICT data_in, double * RESTRICT data_out,
    const TensorIdx stride_in_outld, const TensorIdx stride_out_inld) const;

template <>
KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_HALF>::KernelTrans();
template <>
void KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_HALF>::exec(
    const FloatComplex * RESTRICT data_in, FloatComplex * RESTRICT data_out,
    const TensorIdx stride_in_outld, const TensorIdx stride_out_inld) const;

template <>
KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_HALF>::KernelTrans();
template <>
void KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_HALF>::exec(
    const DoubleComplex * RESTRICT data_in, DoubleComplex * RESTRICT data_out,
    const TensorIdx stride_in_outld, const TensorIdx stride_out_inld) const;


/*
 * Specialization of class KernelTrans, linear kernel, used for common leading
 */
template <typename FloatType>
class KernelTrans<FloatType, KernelTypeTrans::KERNEL_LINE>
    : public KernelTransData<FloatType, KernelTypeTrans::KERNEL_LINE> {
public:
  static constexpr TensorUInt LOOP_MAX = 10;

  KernelTrans();

  void set_wrapper_loop(const TensorIdx stride_in_inld,
      const TensorIdx stride_in_outld, const TensorIdx stride_out_inld,
      const TensorIdx stride_out_outld, const TensorUInt size_kn_inld,
      const TensorUInt size_kn_outld);

  void exec(const FloatType * RESTRICT data_in, FloatType * RESTRICT data_out,
      const TensorIdx size_trans, const TensorIdx size_pad) const;

private:
  TensorIdx stride_in_inld_, stride_in_outld_, stride_out_inld_,
      stride_out_outld_;
  TensorUInt size_kn_inld_, size_kn_outld_;
};


/*
 * Explicit template instantiation declaration for class KernelTrans
 */
extern template class KernelTrans<float, KernelTypeTrans::KERNEL_FULL>;
extern template class KernelTrans<double, KernelTypeTrans::KERNEL_FULL>;
extern template class KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_FULL>;
extern template class KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_FULL>;

extern template class KernelTrans<float, KernelTypeTrans::KERNEL_HALF>;
extern template class KernelTrans<double, KernelTypeTrans::KERNEL_HALF>;
extern template class KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_HALF>;
extern template class KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_HALF>;

extern template class KernelTrans<float, KernelTypeTrans::KERNEL_LINE>;
extern template class KernelTrans<double, KernelTypeTrans::KERNEL_LINE>;
extern template class KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_LINE>;
extern template class KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_LINE>;

#endif // HPTC_ARCH_AVX_KERNEL_TRANS_AVX_TCC_
