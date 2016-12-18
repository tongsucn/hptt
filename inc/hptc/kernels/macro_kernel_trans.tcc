#pragma once
#ifndef HPTC_KERNELS_MACRO_KERNEL_TRANS_TCC_
#define HPTC_KERNELS_MACRO_KERNEL_TRANS_TCC_

template <typename FloatType,
          typename KernelFunc>
MacroTransData<FloatType, KernelFunc>::MacroTransData(KernelFunc kernel,
    DeducedFloatType<FloatType> alpha, DeducedFloatType<FloatType> beta)
    : kernel_(kernel),
      reg_alpha_(reg_coef(alpha)),
      reg_beta_(reg_coef(beta)) {
  template <CoefUsage USAGE>
  using FullK = decltype(&kernel_trans_full<FloatType, USAGE>);
  constexpr bool use_full
      = std::is_same<FullK<CoefUsage::USE_NONE>, decltype(kernel)>::value or
      std::is_same<FullK<CoefUsage::USE_ALPHA>, decltype(kernel)>::value or
      std::is_same<FullK<CoefUsage::USE_BETA>, decltype(kernel)>::value or
      std::is_same<FullK<CoefUsage::USE_BOTH>, decltype(kernel)>::value;

  if (use_full)
    this->reg_num_ = reg_num_full<FloatType>();
  else
    this->reg_num_ = reg_num_half<FloatType>();
}


template <typename FloatType,
          typename KernelFunc,
          MemLayout LAYOUT>
class MacroTrans<FloatType, KernelFunc, LAYOUT, 1, 1>
    : public MacroTransData<FloatType, KernelFunc> {
public:
  MacroTrans(KernelFunc kernel, DeducedFloatType<FloatType> alpha,
      DeducedFloatType<FloatType> beta)
      : MacroTransData<FloatType, KernelFunc>(kernel, alpha, beta) {
  }

  INLINE void operator()(const FloatType * RESTRICT input_data,
      FloatType * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride) {
    this->kernel_(input_data, output_data, input_stride, output_stride,
        this->reg_alpha_, this->reg_beta_);
  }
};


template <typename FloatType,
          typename KernelFunc,
          MemLayout LAYOUT>
class MacroTrans<FloatType, KernelFunc, LAYOUT, 1, 2>
    : public MacroTransData<FloatType, KernelFunc> {
public:
  MacroTrans(KernelFunc kernel, DeducedFloatType<FloatType> alpha,
      DeducedFloatType<FloatType> beta)
      : MacroTransData<FloatType, KernelFunc>(kernel, alpha, beta) {
  }

  INLINE void operator()(const FloatType * RESTRICT input_data,
      FloatType * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride) {
    // Tiling kernel according to memory layout
    constexpr bool is_col_major = MemLayout::COL_MAJOR == LAYOUT;
    if (is_col_major) {
      this->kernel_(input_data, output_data, input_stride, output_stride,
          this->reg_alpha_, this->reg_beta_);
      this->kernel_(input_data + this->reg_num_ * input_stride,
          output_data + this->reg_num_,
          input_stride, output_stride, this->reg_alpha_, this->reg_beta_);
    }
    else {
      this->kernel_(input_data, output_data, input_stride, output_stride,
          this->reg_alpha_, this->reg_beta_);
      this->kernel_(input_data + this->reg_num_,
          output_data + this->reg_num_ * output_stride,
          input_stride, output_stride, this->reg_alpha_, this->reg_beta_);
    }
  }
};


template <typename FloatType,
          typename KernelFunc,
          MemLayout LAYOUT>
class MacroTrans<FloatType, KernelFunc, LAYOUT, 2, 1>
    : public MacroTransData<FloatType, KernelFunc> {
public:
  MacroTrans(KernelFunc kernel, DeducedFloatType<FloatType> alpha,
      DeducedFloatType<FloatType> beta)
      : MacroTransData<FloatType, KernelFunc>(kernel, alpha, beta) {
  }

  INLINE void operator()(const FloatType * RESTRICT input_data,
      FloatType * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride) {
    // Tiling kernel according to memory layout
    constexpr bool is_col_major = MemLayout::COL_MAJOR == LAYOUT;
    if (is_col_major) {
      this->kernel_(input_data, output_data, input_stride, output_stride,
          this->reg_alpha_, this->reg_beta_);
      this->kernel_(input_data + this->reg_num_,
          output_data + this->reg_num_ * output_stride,
          input_stride, output_stride, this->reg_alpha_, this->reg_beta_);
    }
    else {
      this->kernel_(input_data, output_data, input_stride, output_stride,
          this->reg_alpha_, this->reg_beta_);
      this->kernel_(input_data + this->reg_num_ * input_stride,
          output_data + this->reg_num_,
          input_stride, output_stride, this->reg_alpha_, this->reg_beta_);
    }
  }
};


template <typename FloatType,
          typename KernelFunc,
          MemLayout LAYOUT>
class MacroTrans<FloatType, KernelFunc, LAYOUT, 2, 2>
    : public MacroTransData<FloatType, KernelFunc> {
public:
  MacroTrans(KernelFunc kernel, DeducedFloatType<FloatType> alpha,
      DeducedFloatType<FloatType> beta)
      : MacroTransData<FloatType, KernelFunc>(kernel, alpha, beta) {
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
    this->kernel_(input_data + this->reg_num + this->reg_num_ * input_stride,
        output_data + this->reg_num_ + this->reg_num_ * output_stride,
        input_stride, output_stride, this->reg_alpha_, this->reg_beta_);
  }
};


template <typename FloatType,
          typename KernelFunc,
          MemLayout LAYOUT>
class MacroTrans<FloatType, KernelFunc, LAYOUT, 3, 3>
    : public MacroTransData<FloatType, KernelFunc> {
public:
  MacroTrans(KernelFunc kernel, DeducedFloatType<FloatType> alpha,
      DeducedFloatType<FloatType> beta)
      : MacroTransData<FloatType, KernelFunc>(kernel, alpha, beta) {
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
        input_data + this->reg_num + this->reg_num_ * input_stride,
        output_data + this->reg_num_ + this->reg_num_ * output_stride,
        input_stride, output_stride, this->reg_alpha_, this->reg_beta_);
    this->kernel_(
        input_data + this->reg_num + 2 * this->reg_num_ * input_stride,
        output_data + 2 * this->reg_num_ + this->reg_num_ * output_stride,
        input_stride, output_stride, this->reg_alpha_, this->reg_beta_);

    // Third non-continuous memory column
    this->kernel_(
        input_data + 2 * this->reg_num_,
        output_data + 2 * this->reg_num_ * output_stride,
        input_stride, output_stride, this->reg_alpha_, this->reg_beta_);
    this->kernel_(
        input_data + 2 * this->reg_num + this->reg_num_ * input_stride,
        output_data + this->reg_num_ + 2 * this->reg_num_ * output_stride,
        input_stride, output_stride, this->reg_alpha_, this->reg_beta_);
    this->kernel_(
        input_data + 2 * this->reg_num + 2 * this->reg_num_ * input_stride,
        output_data + 2 * this->reg_num_ + 2 * this->reg_num_ * output_stride,
        input_stride, output_stride, this->reg_alpha_, this->reg_beta_);
  }
};


template <typename FloatType,
          typename KernelFunc,
          MemLayout LAYOUT>
class MacroTrans<FloatType, KernelFunc, LAYOUT, 4, 4>
    : public MacroTransData<FloatType, KernelFunc> {
public:
  MacroTrans(KernelFunc kernel, DeducedFloatType<FloatType> alpha,
      DeducedFloatType<FloatType> beta)
      : MacroTransData<FloatType, KernelFunc>(kernel, alpha, beta) {
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
        input_data + this->reg_num + this->reg_num_ * input_stride,
        output_data + this->reg_num_ + this->reg_num_ * output_stride,
        input_stride, output_stride, this->reg_alpha_, this->reg_beta_);
    this->kernel_(
        input_data + this->reg_num + 2 * this->reg_num_ * input_stride,
        output_data + 2 * this->reg_num_ + this->reg_num_ * output_stride,
        input_stride, output_stride, this->reg_alpha_, this->reg_beta_);
    this->kernel_(
        input_data + this->reg_num + 3 * this->reg_num_ * input_stride,
        output_data + 3 * this->reg_num_ + this->reg_num_ * output_stride,
        input_stride, output_stride, this->reg_alpha_, this->reg_beta_);

    // Third non-continuous memory column
    this->kernel_(
        input_data + 2 * this->reg_num_,
        output_data + 2 * this->reg_num_ * output_stride,
        input_stride, output_stride, this->reg_alpha_, this->reg_beta_);
    this->kernel_(
        input_data + 2 * this->reg_num + this->reg_num_ * input_stride,
        output_data + this->reg_num_ + 2 * this->reg_num_ * output_stride,
        input_stride, output_stride, this->reg_alpha_, this->reg_beta_);
    this->kernel_(
        input_data + 2 * this->reg_num + 2 * this->reg_num_ * input_stride,
        output_data + 2 * this->reg_num_ + 2 * this->reg_num_ * output_stride,
        input_stride, output_stride, this->reg_alpha_, this->reg_beta_);
    this->kernel_(
        input_data + 2 * this->reg_num + 3 * this->reg_num_ * input_stride,
        output_data + 3 * this->reg_num_ + 2 * this->reg_num_ * output_stride,
        input_stride, output_stride, this->reg_alpha_, this->reg_beta_);

    // Forth non-continuous memory column
    this->kernel_(
        input_data + 3 * this->reg_num_,
        output_data + 3 * this->reg_num_ * output_stride,
        input_stride, output_stride, this->reg_alpha_, this->reg_beta_);
    this->kernel_(
        input_data + 3 * this->reg_num + this->reg_num_ * input_stride,
        output_data + this->reg_num_ + 3 * this->reg_num_ * output_stride,
        input_stride, output_stride, this->reg_alpha_, this->reg_beta_);
    this->kernel_(
        input_data + 3 * this->reg_num + 2 * this->reg_num_ * input_stride,
        output_data + 2 * this->reg_num_ + 3 * this->reg_num_ * output_stride,
        input_stride, output_stride, this->reg_alpha_, this->reg_beta_);
    this->kernel_(
        input_data + 3 * this->reg_num + 3 * this->reg_num_ * input_stride,
        output_data + 3 * this->reg_num_ + 3 * this->reg_num_ * output_stride,
        input_stride, output_stride, this->reg_alpha_, this->reg_beta_);
  }
};

#endif // HPTC_KERNELS_MACRO_KERNEL_TRANS_TCC_
