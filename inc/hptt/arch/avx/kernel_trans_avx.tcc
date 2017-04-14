#pragma once
#ifndef HPTT_ARCH_AVX_KERNEL_TRANS_AVX_TCC_
#define HPTT_ARCH_AVX_KERNEL_TRANS_AVX_TCC_

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

  static HPTT_INL Reg set_reg(const Deduced coef) {
    return _mm256_set1_ps(coef);
  }
  static HPTT_INL Reg load(const FloatType *target) {
    return _mm256_loadu_ps(reinterpret_cast<const Deduced *>(target));
  }
  static HPTT_INL void store(FloatType *target, const Reg &reg) {
    _mm256_storeu_ps(reinterpret_cast<Deduced *>(target), reg);
  }
  static HPTT_INL void stream(FloatType *data_out, const Reg &reg) {
    _mm256_stream_ps(reinterpret_cast<Deduced *>(data_out), reg);
  }
  static HPTT_INL void sstore(FloatType *data_out, const FloatType *buffer) {
    IntrinImpl::stream(data_out, IntrinImpl::load(buffer));
    if (2 == sizeof(FloatType) / sizeof(Deduced))
      _mm256_stream_ps(reinterpret_cast<Deduced *>(data_out) + 1,
          _mm256_load_ps(reinterpret_cast<const Deduced *>(buffer) + 1));
  }
  static HPTT_INL Reg add(const Reg &reg_a, const Reg &reg_b) {
    return _mm256_add_ps(reg_a, reg_b);
  }
  static HPTT_INL Reg mul(const Reg &reg_a, const Reg &reg_b) {
    return _mm256_mul_ps(reg_a, reg_b);
  }
};

template <typename FloatType,
          KernelTypeTrans TYPE>
struct IntrinImpl<FloatType, TYPE,
    Enable<TypeSelector<FloatType, TYPE>::fl_dz>> {
  using Deduced = DeducedFloatType<FloatType>;
  using Reg = RegType<FloatType, TYPE>;

  static HPTT_INL Reg set_reg(const Deduced coef) {
    return _mm256_set1_pd(coef);
  }
  static HPTT_INL Reg load(const FloatType *target) {
    return _mm256_loadu_pd(reinterpret_cast<const Deduced *>(target));
  }
  static HPTT_INL void store(FloatType *target, const Reg &reg) {
    _mm256_storeu_pd(reinterpret_cast<Deduced *>(target), reg);
  }
  static HPTT_INL void stream(FloatType *data_out, const Reg &reg) {
    _mm256_stream_pd(reinterpret_cast<Deduced *>(data_out), reg);
  }
  static HPTT_INL void sstore(FloatType *data_out, const FloatType *buffer) {
    IntrinImpl::stream(data_out, IntrinImpl::load(buffer));
    if (2 == sizeof(FloatType) / sizeof(Deduced))
      _mm256_stream_pd(reinterpret_cast<Deduced *>(data_out) + 1,
          _mm256_load_pd(reinterpret_cast<const Deduced *>(buffer) + 1));
  }
  static HPTT_INL Reg add(const Reg &reg_a, const Reg &reg_b) {
    return _mm256_add_pd(reg_a, reg_b);
  }
  static HPTT_INL Reg mul(const Reg &reg_a, const Reg &reg_b) {
    return _mm256_mul_pd(reg_a, reg_b);
  }
};

template <typename FloatType,
          KernelTypeTrans TYPE>
struct IntrinImpl<FloatType, TYPE,
    Enable<TypeSelector<FloatType, TYPE>::h_sc>> {
  using Deduced = DeducedFloatType<FloatType>;
  using Reg = RegType<FloatType, TYPE>;

  static HPTT_INL RegType<FloatType, TYPE> set_reg(const Deduced coef) {
    return _mm_set1_ps(coef);
  }
  static HPTT_INL Reg load(const FloatType *target) {
    return _mm_loadu_ps(reinterpret_cast<const Deduced *>(target));
  }
  static HPTT_INL void store(FloatType *target, const Reg &reg) {
    _mm_storeu_ps(reinterpret_cast<Deduced *>(target), reg);
  }
  static HPTT_INL void sstore(FloatType *data_out, const FloatType *buffer) {
    _mm_stream_ps(reinterpret_cast<Deduced *>(data_out),
        _mm_load_ps(reinterpret_cast<const Deduced *>(buffer)));
    if (2 == sizeof(FloatType) / sizeof(Deduced))
      _mm_stream_ps(reinterpret_cast<Deduced *>(data_out) + 1,
          _mm_load_ps(reinterpret_cast<const Deduced *>(buffer) + 1));
  }
  static HPTT_INL Reg add(const Reg &reg_a, const Reg &reg_b) {
    return _mm_add_ps(reg_a, reg_b);
  }
  static HPTT_INL Reg mul(const Reg &reg_a, const Reg &reg_b) {
    return _mm_mul_ps(reg_a, reg_b);
  }
};

template <typename FloatType,
          KernelTypeTrans TYPE>
struct IntrinImpl<FloatType, TYPE, Enable<TypeSelector<FloatType, TYPE>::h_d>> {
  using Reg = RegType<FloatType, TYPE>;

  static HPTT_INL RegType<FloatType, TYPE> set_reg(const FloatType coef) {
    return _mm_set1_pd(coef);
  }
  static HPTT_INL Reg load(const FloatType *target) {
    return _mm_loadu_pd(target);
  }
  static HPTT_INL void store(FloatType *target, const Reg &reg) {
    _mm_storeu_pd(target, reg);
  }
  static HPTT_INL void sstore(FloatType *data_out, const FloatType *buffer) {
    _mm_stream_pd(data_out, _mm_load_pd(buffer));
  }
  static HPTT_INL Reg add(const Reg &reg_a, const Reg &reg_b) {
    return _mm_add_pd(reg_a, reg_b);
  }
  static HPTT_INL Reg mul(const Reg &reg_a, const Reg &reg_b) {
    return _mm_mul_pd(reg_a, reg_b);
  }
};

template <typename FloatType,
          KernelTypeTrans TYPE>
struct IntrinImpl<FloatType, TYPE, Enable<TypeSelector<FloatType, TYPE>::h_z>> {
  using Deduced = DeducedFloatType<FloatType>;
  using Reg = RegType<FloatType, TYPE>;

  static HPTT_INL RegType<FloatType, TYPE> set_reg(const Deduced coef) {
    return coef;
  }
  static HPTT_INL void sstore(FloatType *data_out, const FloatType *buffer) {
    *data_out = *buffer;
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
void KernelTransData<FloatType, TYPE>::sstore(FloatType *data_out,
    const FloatType *buffer) {
  IntrinImpl<FloatType, TYPE>::sstore(data_out, buffer);
}


template <typename FloatType,
          KernelTypeTrans TYPE>
bool KernelTransData<FloatType, TYPE>::check_stream(TensorUInt arr_size) {
  return (arr_size * sizeof(FloatType)) % 64 == 0;
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
 * Partial pecializations of class KernelTrans
 */
template <bool UPDATE_OUT>
class KernelTrans<float, KernelTypeTrans::KERNEL_FULL, UPDATE_OUT>
    : public KernelTransData<float, KernelTypeTrans::KERNEL_FULL> {
public:
  static constexpr bool UPDATE = UPDATE_OUT;
  KernelTrans();
  void exec(const float * RESTRICT data_in, float * RESTRICT data_out,
      const TensorIdx stride_in_outld, const TensorIdx stride_out_inld) const;
};

template <bool UPDATE_OUT>
class KernelTrans<float, KernelTypeTrans::KERNEL_HALF, UPDATE_OUT>
    : public KernelTransData<float, KernelTypeTrans::KERNEL_HALF> {
public:
  static constexpr bool UPDATE = UPDATE_OUT;
  KernelTrans();
  void exec(const float * RESTRICT data_in, float * RESTRICT data_out,
      const TensorIdx stride_in_outld, const TensorIdx stride_out_inld) const;
};

template <bool UPDATE_OUT>
class KernelTrans<double, KernelTypeTrans::KERNEL_FULL, UPDATE_OUT>
    : public KernelTransData<double, KernelTypeTrans::KERNEL_FULL> {
public:
  static constexpr bool UPDATE = UPDATE_OUT;
  KernelTrans();
  void exec(const double * RESTRICT data_in, double * RESTRICT data_out,
      const TensorIdx stride_in_outld, const TensorIdx stride_out_inld) const;
};

template <bool UPDATE_OUT>
class KernelTrans<double, KernelTypeTrans::KERNEL_HALF, UPDATE_OUT>
    : public KernelTransData<double, KernelTypeTrans::KERNEL_HALF> {
public:
  static constexpr bool UPDATE = UPDATE_OUT;
  KernelTrans();
  void exec(const double * RESTRICT data_in, double * RESTRICT data_out,
      const TensorIdx stride_in_outld, const TensorIdx stride_out_inld) const;
};

template <bool UPDATE_OUT>
class KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_FULL, UPDATE_OUT>
    : public KernelTransData<FloatComplex, KernelTypeTrans::KERNEL_FULL> {
public:
  static constexpr bool UPDATE = UPDATE_OUT;
  KernelTrans();
  void exec(const FloatComplex * RESTRICT data_in,
      FloatComplex * RESTRICT data_out, const TensorIdx stride_in_outld,
      const TensorIdx stride_out_inld) const;
};

template <bool UPDATE_OUT>
class KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_HALF, UPDATE_OUT>
    : public KernelTransData<FloatComplex, KernelTypeTrans::KERNEL_HALF> {
public:
  static constexpr bool UPDATE = UPDATE_OUT;
  KernelTrans();
  void exec(const FloatComplex * RESTRICT data_in,
      FloatComplex * RESTRICT data_out, const TensorIdx stride_in_outld,
      const TensorIdx stride_out_inld) const;
};

template <bool UPDATE_OUT>
class KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_FULL, UPDATE_OUT>
    : public KernelTransData<DoubleComplex, KernelTypeTrans::KERNEL_FULL> {
public:
  static constexpr bool UPDATE = UPDATE_OUT;
  KernelTrans();
  void exec(const DoubleComplex * RESTRICT data_in,
      DoubleComplex * RESTRICT data_out, const TensorIdx stride_in_outld,
      const TensorIdx stride_out_inld) const;
};

template <bool UPDATE_OUT>
class KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_HALF, UPDATE_OUT>
    : public KernelTransData<DoubleComplex, KernelTypeTrans::KERNEL_HALF> {
public:
  static constexpr bool UPDATE = UPDATE_OUT;
  KernelTrans();
  void exec(const DoubleComplex * RESTRICT data_in,
      DoubleComplex * RESTRICT data_out, const TensorIdx stride_in_outld,
      const TensorIdx stride_out_inld) const;
};


/*
 * Specialization of class KernelTrans, linear kernel, used for common leading
 */
template <typename FloatType,
          bool UPDATE_OUT>
class KernelTrans<FloatType, KernelTypeTrans::KERNEL_LINE, UPDATE_OUT>
    : public KernelTransData<FloatType, KernelTypeTrans::KERNEL_LINE> {
public:
  KernelTrans();

  void exec(const FloatType * RESTRICT data_in, FloatType * RESTRICT data_out,
      const TensorIdx size_trans, const TensorIdx) const;
};


/*
 * Explicit template instantiation declaration for class KernelTrans
 */
extern template class KernelTrans<float, KernelTypeTrans::KERNEL_FULL, true>;
extern template class KernelTrans<double, KernelTypeTrans::KERNEL_FULL, true>;
extern template class KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_FULL,
    true>;
extern template class KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_FULL,
    true>;

extern template class KernelTrans<float, KernelTypeTrans::KERNEL_HALF, true>;
extern template class KernelTrans<double, KernelTypeTrans::KERNEL_HALF, true>;
extern template class KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_HALF,
    true>;
extern template class KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_HALF,
    true>;

extern template class KernelTrans<float, KernelTypeTrans::KERNEL_LINE, true>;
extern template class KernelTrans<double, KernelTypeTrans::KERNEL_LINE, true>;
extern template class KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_LINE,
    true>;
extern template class KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_LINE,
    true>;

extern template class KernelTrans<float, KernelTypeTrans::KERNEL_FULL, false>;
extern template class KernelTrans<double, KernelTypeTrans::KERNEL_FULL, false>;
extern template class KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_FULL,
    false>;
extern template class KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_FULL,
    false>;

extern template class KernelTrans<float, KernelTypeTrans::KERNEL_HALF, false>;
extern template class KernelTrans<double, KernelTypeTrans::KERNEL_HALF, false>;
extern template class KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_HALF,
    false>;
extern template class KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_HALF,
    false>;

extern template class KernelTrans<float, KernelTypeTrans::KERNEL_LINE, false>;
extern template class KernelTrans<double, KernelTypeTrans::KERNEL_LINE, false>;
extern template class KernelTrans<FloatComplex, KernelTypeTrans::KERNEL_LINE,
    false>;
extern template class KernelTrans<DoubleComplex, KernelTypeTrans::KERNEL_LINE,
    false>;

#endif // HPTT_ARCH_AVX_KERNEL_TRANS_AVX_TCC_
