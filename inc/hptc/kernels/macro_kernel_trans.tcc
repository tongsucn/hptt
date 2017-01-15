#pragma once
#ifndef HPTC_KERNELS_MACRO_KERNEL_TRANS_TCC_
#define HPTC_KERNELS_MACRO_KERNEL_TRANS_TCC_

/*
 * Implementation for class MacroTransVecData
 */
template <typename FloatType,
          typename KernelFunc,
          GenNumType CONT_LEN,
          GenNumType NCONT_LEN>
MacroTransVecData<FloatType, KernelFunc, CONT_LEN, NCONT_LEN>
::MacroTransVecData(
    KernelFunc kernel, DeducedFloatType<FloatType> alpha,
    DeducedFloatType<FloatType> beta)
    : kernel_(kernel),
      reg_alpha_(kernel.reg_coef(alpha)),
      reg_beta_(kernel.reg_coef(beta)),
      reg_num_(kernel.get_reg_num()) {
}


template <typename FloatType,
          typename KernelFunc,
          GenNumType CONT_LEN,
          GenNumType NCONT_LEN>
INLINE GenNumType
MacroTransVecData<FloatType, KernelFunc, CONT_LEN, NCONT_LEN>::get_cont_len() {
  return CONT_LEN * this->reg_num_;
}


template <typename FloatType,
          typename KernelFunc,
          GenNumType CONT_LEN,
          GenNumType NCONT_LEN>
INLINE GenNumType
MacroTransVecData<FloatType, KernelFunc, CONT_LEN, NCONT_LEN>::get_ncont_len() {
  return NCONT_LEN * this->reg_num_;
}


template <typename FloatType,
          typename KernelFunc,
          GenNumType CONT_LEN,
          GenNumType NCONT_LEN>
template <GenNumType CONT,
         GenNumType NCONT>
INLINE void MacroTransVecData<FloatType, KernelFunc, CONT_LEN, NCONT_LEN>::
ncont_tiler(DualCounter<CONT, NCONT>,
    const FloatType * RESTRICT input_data, FloatType * RESTRICT output_data,
    const TensorIdx input_stride, const TensorIdx output_stride) {
  this->ncont_tiler(DualCounter<CONT, NCONT - 1>(), input_data, output_data,
      input_stride, output_stride);
  this->cont_tiler(DualCounter<CONT - 1, NCONT - 1>(), input_data, output_data,
      input_stride, output_stride);
}


template <typename FloatType,
          typename KernelFunc,
          GenNumType CONT_LEN,
          GenNumType NCONT_LEN>
template <GenNumType CONT>
INLINE void MacroTransVecData<FloatType, KernelFunc, CONT_LEN, NCONT_LEN>::
ncont_tiler(DualCounter<CONT, 0>,
    const FloatType * RESTRICT input_data, FloatType * RESTRICT output_data,
    const TensorIdx input_stride, const TensorIdx output_stride) {
}


template <typename FloatType,
          typename KernelFunc,
          GenNumType CONT_LEN,
          GenNumType NCONT_LEN>
template <GenNumType CONT,
         GenNumType NCONT>
INLINE void MacroTransVecData<FloatType, KernelFunc, CONT_LEN, NCONT_LEN>::
cont_tiler(DualCounter<CONT, NCONT>,
    const FloatType * RESTRICT input_data, FloatType * RESTRICT output_data,
    const TensorIdx input_stride, const TensorIdx output_stride) {
  this->cont_tiler(DualCounter<CONT - 1, NCONT>(), input_data, output_data,
      input_stride, output_stride);
  this->kernel_(input_data + CONT * reg_num_ + NCONT * reg_num_ * input_stride,
      output_data + NCONT * reg_num_ + CONT * reg_num_ * output_stride,
      input_stride, output_stride, this->reg_alpha_, this->reg_beta_);
}


template <typename FloatType,
          typename KernelFunc,
          GenNumType CONT_LEN,
          GenNumType NCONT_LEN>
template <GenNumType NCONT>
INLINE void MacroTransVecData<FloatType, KernelFunc, CONT_LEN, NCONT_LEN>::
cont_tiler(DualCounter<0, NCONT>,
    const FloatType * RESTRICT input_data, FloatType * RESTRICT output_data,
    const TensorIdx input_stride, const TensorIdx output_stride) {
  this->kernel_(input_data + NCONT * this->reg_num_ * input_stride,
      output_data + NCONT * this->reg_num_, input_stride, output_stride,
      this->reg_alpha_, this->reg_beta_);
}


/*
 * Implementation for class MacroTransVec
 */
template <typename FloatType,
          typename KernelFunc>
class MacroTransVec<FloatType, KernelFunc, 0, 0>
    : public MacroTransVecData<FloatType, KernelFunc, 0, 0> {
};


/*template <typename FloatType,
          typename KernelFunc>
class MacroTransVec<FloatType, KernelFunc, 4, 4>
    : public MacroTransVecData<FloatType, KernelFunc, 4, 4> {
public:
  MacroTransVec(KernelFunc kernel, DeducedFloatType<FloatType> alpha,
      DeducedFloatType<FloatType> beta)
      : MacroTransVecData<FloatType, KernelFunc, 4, 4>(kernel, alpha, beta) {
  }

  INLINE void operator()(const FloatType * RESTRICT input_data,
      FloatType * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride) {
    // First non-continuous memory column
    this->kernel_(input_data, output_data, input_stride, output_stride,
        this->reg_alpha_, this->reg_beta_);
    this->kernel_(input_data + this->reg_num_ * input_stride,
        output_data + this->reg_num_,
        input_stride, output_stride, this->reg_alpha_, this->reg_beta_);
    this->kernel_(input_data + 2 * this->reg_num_ * input_stride,
        output_data + 2 * this->reg_num_,
        input_stride, output_stride, this->reg_alpha_, this->reg_beta_);
    this->kernel_(input_data + 3 * this->reg_num_ * input_stride,
        output_data + 3 * this->reg_num_,
        input_stride, output_stride, this->reg_alpha_, this->reg_beta_);

    // Second non-continuous memory column
    this->kernel_(
        input_data + this->reg_num_,
        output_data + this->reg_num_ * output_stride,
        input_stride, output_stride, this->reg_alpha_, this->reg_beta_);
    this->kernel_(
        input_data + this->reg_num_ + this->reg_num_ * input_stride,
        output_data + this->reg_num_ + this->reg_num_ * output_stride,
        input_stride, output_stride, this->reg_alpha_, this->reg_beta_);
    this->kernel_(
        input_data + this->reg_num_ + 2 * this->reg_num_ * input_stride,
        output_data + 2 * this->reg_num_ + this->reg_num_ * output_stride,
        input_stride, output_stride, this->reg_alpha_, this->reg_beta_);
    this->kernel_(
        input_data + this->reg_num_ + 3 * this->reg_num_ * input_stride,
        output_data + 3 * this->reg_num_ + this->reg_num_ * output_stride,
        input_stride, output_stride, this->reg_alpha_, this->reg_beta_);

    // Third non-continuous memory column
    this->kernel_(
        input_data + 2 * this->reg_num_,
        output_data + 2 * this->reg_num_ * output_stride,
        input_stride, output_stride, this->reg_alpha_, this->reg_beta_);
    this->kernel_(
        input_data + 2 * this->reg_num_ + this->reg_num_ * input_stride,
        output_data + this->reg_num_ + 2 * this->reg_num_ * output_stride,
        input_stride, output_stride, this->reg_alpha_, this->reg_beta_);
    this->kernel_(
        input_data + 2 * this->reg_num_ + 2 * this->reg_num_ * input_stride,
        output_data + 2 * this->reg_num_ + 2 * this->reg_num_ * output_stride,
        input_stride, output_stride, this->reg_alpha_, this->reg_beta_);
    this->kernel_(
        input_data + 2 * this->reg_num_ + 3 * this->reg_num_ * input_stride,
        output_data + 3 * this->reg_num_ + 2 * this->reg_num_ * output_stride,
        input_stride, output_stride, this->reg_alpha_, this->reg_beta_);

    // Forth non-continuous memory column
    this->kernel_(
        input_data + 3 * this->reg_num_,
        output_data + 3 * this->reg_num_ * output_stride,
        input_stride, output_stride, this->reg_alpha_, this->reg_beta_);
    this->kernel_(
        input_data + 3 * this->reg_num_ + this->reg_num_ * input_stride,
        output_data + this->reg_num_ + 3 * this->reg_num_ * output_stride,
        input_stride, output_stride, this->reg_alpha_, this->reg_beta_);
    this->kernel_(
        input_data + 3 * this->reg_num_ + 2 * this->reg_num_ * input_stride,
        output_data + 2 * this->reg_num_ + 3 * this->reg_num_ * output_stride,
        input_stride, output_stride, this->reg_alpha_, this->reg_beta_);
    this->kernel_(
        input_data + 3 * this->reg_num_ + 3 * this->reg_num_ * input_stride,
        output_data + 3 * this->reg_num_ + 3 * this->reg_num_ * output_stride,
        input_stride, output_stride, this->reg_alpha_, this->reg_beta_);
  }
};*/


template <typename FloatType,
          typename KernelFunc,
          GenNumType CONT_LEN,
          GenNumType NCONT_LEN>
MacroTransVec<FloatType, KernelFunc, CONT_LEN, NCONT_LEN>::MacroTransVec(
    KernelFunc kernel, DeducedFloatType<FloatType> alpha,
    DeducedFloatType<FloatType> beta)
    : MacroTransVecData<FloatType, KernelFunc, CONT_LEN, NCONT_LEN>(kernel,
        alpha, beta) {
}


template <typename FloatType,
          typename KernelFunc,
          GenNumType CONT_LEN,
          GenNumType NCONT_LEN>
INLINE void MacroTransVec<FloatType, KernelFunc, CONT_LEN, NCONT_LEN>::
operator()(const FloatType * RESTRICT input_data,
    FloatType * RESTRICT output_data, const TensorIdx input_stride,
    const TensorIdx output_stride) {
  this->ncont_tiler(DualCounter<CONT_LEN, NCONT_LEN>(),
      input_data, output_data, input_stride, output_stride);
}


/*
 * Implementation for class MacroTransScalarData
 */
template <typename FloatType,
          CoefUsage USAGE>
MacroTransScalarData<FloatType, USAGE>::MacroTransScalarData(
    DeducedFloatType<FloatType> alpha, DeducedFloatType<FloatType> beta)
    : alpha(alpha), beta(beta) {
}


/*
 * Implementation for class MacroTransScalar
 */
template <typename FloatType>
class MacroTransScalar<FloatType, CoefUsage::USE_NONE>
    : public MacroTransScalarData<FloatType, CoefUsage::USE_NONE> {
public:
  MacroTransScalar(DeducedFloatType<FloatType> alpha,
      DeducedFloatType<FloatType> beta)
      : MacroTransScalarData<FloatType, CoefUsage::USE_NONE>(alpha, beta) {
  }

  INLINE void operator()(const FloatType &input_data, FloatType &output_data) {
    output_data = input_data;
  }
};


template <typename FloatType>
class MacroTransScalar<FloatType, CoefUsage::USE_ALPHA>
    : public MacroTransScalarData<FloatType, CoefUsage::USE_ALPHA> {
public:
  MacroTransScalar(DeducedFloatType<FloatType> alpha,
      DeducedFloatType<FloatType> beta)
      : MacroTransScalarData<FloatType, CoefUsage::USE_ALPHA>(alpha, beta) {
  }

  INLINE void operator()(const FloatType &input_data, FloatType &output_data) {
    output_data = this->alpha * input_data;
  }
};


template <typename FloatType>
class MacroTransScalar<FloatType, CoefUsage::USE_BETA>
    : public MacroTransScalarData<FloatType, CoefUsage::USE_BETA> {
public:
  MacroTransScalar(DeducedFloatType<FloatType> alpha,
      DeducedFloatType<FloatType> beta)
      : MacroTransScalarData<FloatType, CoefUsage::USE_BETA>(alpha, beta) {
  }

  INLINE void operator()(const FloatType &input_data, FloatType &output_data) {
    output_data = input_data + this->beta * output_data;
  }
};


template <typename FloatType>
class MacroTransScalar<FloatType, CoefUsage::USE_BOTH>
    : public MacroTransScalarData<FloatType, CoefUsage::USE_BOTH> {
public:
  MacroTransScalar(DeducedFloatType<FloatType> alpha,
      DeducedFloatType<FloatType> beta)
      : MacroTransScalarData<FloatType, CoefUsage::USE_BOTH>(alpha, beta) {
  }

  INLINE void operator()(const FloatType &input_data, FloatType &output_data) {
    output_data = this->alpha * input_data + this->beta * output_data;
  }
};

#endif // HPTC_KERNELS_MACRO_KERNEL_TRANS_TCC_
