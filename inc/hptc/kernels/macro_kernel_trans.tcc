#pragma once
#ifndef HPTC_KERNELS_MACRO_KERNEL_TRANS_TCC_
#define HPTC_KERNELS_MACRO_KERNEL_TRANS_TCC_

template <typename FloatType,
          typename KernelFunc,
          GenNumType CONT_LEN,
          GenNumType NCONT_LEN>
MacroTransData<FloatType, KernelFunc, CONT_LEN, NCONT_LEN>::MacroTransData(
    KernelFunc kernel, DeducedFloatType<FloatType> alpha,
    DeducedFloatType<FloatType> beta)
    : kernel_(kernel),
      reg_alpha_(reg_coef(alpha)),
      reg_beta_(reg_coef(beta)),
      reg_num_(kernel.get_reg_num()) {
}


template <typename FloatType,
          typename KernelFunc,
          GenNumType CONT_LEN,
          GenNumType NCONT_LEN>
INLINE GenNumType
MacroTransData<FloatType, KernelFunc, CONT_LEN, NCONT_LEN>::get_cont_len() {
  return CONT_LEN * this->reg_num_;
}


template <typename FloatType,
          typename KernelFunc,
          GenNumType CONT_LEN,
          GenNumType NCONT_LEN>
INLINE GenNumType
MacroTransData<FloatType, KernelFunc, CONT_LEN, NCONT_LEN>::get_ncont_len() {
  return NCONT_LEN * this->reg_num_;
}


template <typename FloatType,
          typename KernelFunc>
class MacroTrans<FloatType, KernelFunc, 1, 1>
    : public MacroTransData<FloatType, KernelFunc, 1, 1> {
public:
  MacroTrans(KernelFunc kernel, DeducedFloatType<FloatType> alpha,
      DeducedFloatType<FloatType> beta)
      : MacroTransData<FloatType, KernelFunc, 1, 1>(kernel, alpha, beta) {
  }

  INLINE void operator()(const FloatType * RESTRICT input_data,
      FloatType * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride) {
    this->kernel_(input_data, output_data, input_stride, output_stride,
        this->reg_alpha_, this->reg_beta_);
  }
};


template <typename FloatType,
          typename KernelFunc>
class MacroTrans<FloatType, KernelFunc, 1, 2>
    : public MacroTransData<FloatType, KernelFunc, 1, 2> {
public:
  MacroTrans(KernelFunc kernel, DeducedFloatType<FloatType> alpha,
      DeducedFloatType<FloatType> beta)
      : MacroTransData<FloatType, KernelFunc, 1, 2>(kernel, alpha, beta) {
  }

  INLINE void operator()(const FloatType * RESTRICT input_data,
      FloatType * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride) {
    this->kernel_(input_data, output_data, input_stride, output_stride,
        this->reg_alpha_, this->reg_beta_);
    this->kernel_(input_data + this->reg_num_ * input_stride,
        output_data + this->reg_num_,
        input_stride, output_stride, this->reg_alpha_, this->reg_beta_);
  }
};


template <typename FloatType,
          typename KernelFunc>
class MacroTrans<FloatType, KernelFunc, 2, 1>
    : public MacroTransData<FloatType, KernelFunc, 2, 1> {
public:
  MacroTrans(KernelFunc kernel, DeducedFloatType<FloatType> alpha,
      DeducedFloatType<FloatType> beta)
      : MacroTransData<FloatType, KernelFunc, 2, 1>(kernel, alpha, beta) {
  }

  INLINE void operator()(const FloatType * RESTRICT input_data,
      FloatType * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride) {
    this->kernel_(input_data, output_data, input_stride, output_stride,
        this->reg_alpha_, this->reg_beta_);
    this->kernel_(input_data + this->reg_num_,
        output_data + this->reg_num_ * output_stride,
        input_stride, output_stride, this->reg_alpha_, this->reg_beta_);
  }
};


template <typename FloatType,
          typename KernelFunc>
class MacroTrans<FloatType, KernelFunc, 2, 2>
    : public MacroTransData<FloatType, KernelFunc, 2, 2> {
public:
  MacroTrans(KernelFunc kernel, DeducedFloatType<FloatType> alpha,
      DeducedFloatType<FloatType> beta)
      : MacroTransData<FloatType, KernelFunc, 2, 2>(kernel, alpha, beta) {
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

    // Second non-continuous memory column
    this->kernel_(input_data + this->reg_num_,
        output_data + this->reg_num_ * output_stride,
        input_stride, output_stride, this->reg_alpha_, this->reg_beta_);
    this->kernel_(input_data + this->reg_num_ + this->reg_num_ * input_stride,
        output_data + this->reg_num_ + this->reg_num_ * output_stride,
        input_stride, output_stride, this->reg_alpha_, this->reg_beta_);
  }
};


template <typename FloatType,
          typename KernelFunc>
class MacroTrans<FloatType, KernelFunc, 3, 3>
    : public MacroTransData<FloatType, KernelFunc, 3, 3> {
public:
  MacroTrans(KernelFunc kernel, DeducedFloatType<FloatType> alpha,
      DeducedFloatType<FloatType> beta)
      : MacroTransData<FloatType, KernelFunc, 3, 3> (kernel, alpha, beta) {
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
  }
};


template <typename FloatType,
          typename KernelFunc>
class MacroTrans<FloatType, KernelFunc, 4, 4>
    : public MacroTransData<FloatType, KernelFunc, 4, 4> {
public:
  MacroTrans(KernelFunc kernel, DeducedFloatType<FloatType> alpha,
      DeducedFloatType<FloatType> beta)
      : MacroTransData<FloatType, KernelFunc, 4, 4>(kernel, alpha, beta) {
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
};

#endif // HPTC_KERNELS_MACRO_KERNEL_TRANS_TCC_
