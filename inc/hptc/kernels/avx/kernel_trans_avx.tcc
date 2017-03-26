#pragma once
#ifndef HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_TCC_
#define HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_TCC_

/*
 * Partial specialization for class KernelTransAvx
 */
template <CoefUsageTrans USAGE>
struct KernelTransAvx<float, USAGE, KernelTypeTrans::KERNEL_FULL> final
    : public KernelTransAvxBase<float, KernelTypeTrans::KERNEL_FULL> {
  static RegType reg_coef(const DeducedFloatType<float> coef);

  static void exec(const float * RESTRICT input_data,
      float * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride, const RegType &reg_alpha,
      const RegType &reg_beta);
};


template <CoefUsageTrans USAGE>
struct KernelTransAvx<double, USAGE, KernelTypeTrans::KERNEL_FULL> final
    : public KernelTransAvxBase<double, KernelTypeTrans::KERNEL_FULL> {
  static RegType reg_coef(const DeducedFloatType<double> coef);

  static void exec(const double * RESTRICT input_data,
      double * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride, const RegType &reg_alpha,
      const RegType &reg_beta);
};


template <CoefUsageTrans USAGE>
struct KernelTransAvx<FloatComplex, USAGE, KernelTypeTrans::KERNEL_FULL> final
    : public KernelTransAvxBase<FloatComplex, KernelTypeTrans::KERNEL_FULL> {
  static RegType reg_coef(const DeducedFloatType<FloatComplex> coef);

  static void exec(const FloatComplex * RESTRICT input_data,
      FloatComplex * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride, const RegType &reg_alpha,
      const RegType &reg_beta);
};


template <CoefUsageTrans USAGE>
struct KernelTransAvx<DoubleComplex, USAGE, KernelTypeTrans::KERNEL_FULL> final
    : public KernelTransAvxBase<DoubleComplex, KernelTypeTrans::KERNEL_FULL> {
  static RegType reg_coef(const DeducedFloatType<DoubleComplex> coef);

  static void exec(const DoubleComplex * RESTRICT input_data,
      DoubleComplex * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride, const RegType &reg_alpha,
      const RegType &reg_beta);
};


template <CoefUsageTrans USAGE>
struct KernelTransAvx<float, USAGE, KernelTypeTrans::KERNEL_HALF> final
    : public KernelTransAvxBase<float, KernelTypeTrans::KERNEL_HALF> {
  static RegType reg_coef(const DeducedFloatType<float> coef);

  static void exec(const float * RESTRICT input_data,
      float * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride, const RegType &reg_alpha,
      const RegType &reg_beta);
};


template <CoefUsageTrans USAGE>
struct KernelTransAvx<double, USAGE, KernelTypeTrans::KERNEL_HALF> final
    : public KernelTransAvxBase<double, KernelTypeTrans::KERNEL_HALF> {
  static RegType reg_coef(const DeducedFloatType<double> coef);

  static void exec(const double * RESTRICT input_data,
      double * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride, const RegType &reg_alpha,
      const RegType &reg_beta);
};


template <CoefUsageTrans USAGE>
struct KernelTransAvx<FloatComplex, USAGE, KernelTypeTrans::KERNEL_HALF> final
    : public KernelTransAvxBase<FloatComplex, KernelTypeTrans::KERNEL_HALF> {
  static RegType reg_coef(const DeducedFloatType<FloatComplex> coef);

  static void exec(const FloatComplex * RESTRICT input_data,
      FloatComplex * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride, const RegType &reg_alpha,
      const RegType &reg_beta);
};


template <CoefUsageTrans USAGE>
struct KernelTransAvx<DoubleComplex, USAGE, KernelTypeTrans::KERNEL_HALF> final
    : public KernelTransAvxBase<DoubleComplex, KernelTypeTrans::KERNEL_HALF> {
  static RegType reg_coef(const DeducedFloatType<DoubleComplex> coef);

  static void exec(const DoubleComplex * RESTRICT input_data,
      DoubleComplex * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride, const RegType &reg_alpha,
      const RegType &reg_beta);
};


/*
 * Explicit instantiation declaration for struct KernelTransAvx
 */
extern template struct KernelTransAvx<float, CoefUsageTrans::USE_NONE,
    KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvx<float, CoefUsageTrans::USE_NONE,
    KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvx<float, CoefUsageTrans::USE_ALPHA,
    KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvx<float, CoefUsageTrans::USE_ALPHA,
    KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvx<float, CoefUsageTrans::USE_BETA,
    KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvx<float, CoefUsageTrans::USE_BETA,
    KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvx<float, CoefUsageTrans::USE_BOTH,
    KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvx<float, CoefUsageTrans::USE_BOTH,
    KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvx<double, CoefUsageTrans::USE_NONE,
    KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvx<double, CoefUsageTrans::USE_NONE,
    KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvx<double, CoefUsageTrans::USE_ALPHA,
    KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvx<double, CoefUsageTrans::USE_ALPHA,
    KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvx<double, CoefUsageTrans::USE_BETA,
    KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvx<double, CoefUsageTrans::USE_BETA,
    KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvx<double, CoefUsageTrans::USE_BOTH,
    KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvx<double, CoefUsageTrans::USE_BOTH,
    KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvx<FloatComplex, CoefUsageTrans::USE_NONE,
    KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvx<FloatComplex, CoefUsageTrans::USE_NONE,
    KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvx<FloatComplex, CoefUsageTrans::USE_ALPHA,
    KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvx<FloatComplex, CoefUsageTrans::USE_ALPHA,
    KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvx<FloatComplex, CoefUsageTrans::USE_BETA,
    KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvx<FloatComplex, CoefUsageTrans::USE_BETA,
    KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvx<FloatComplex, CoefUsageTrans::USE_BOTH,
    KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvx<FloatComplex, CoefUsageTrans::USE_BOTH,
    KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvx<DoubleComplex, CoefUsageTrans::USE_NONE,
    KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvx<DoubleComplex, CoefUsageTrans::USE_NONE,
    KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvx<DoubleComplex, CoefUsageTrans::USE_ALPHA,
    KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvx<DoubleComplex, CoefUsageTrans::USE_ALPHA,
    KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvx<DoubleComplex, CoefUsageTrans::USE_BETA,
    KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvx<DoubleComplex, CoefUsageTrans::USE_BETA,
    KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvx<DoubleComplex, CoefUsageTrans::USE_BOTH,
    KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvx<DoubleComplex, CoefUsageTrans::USE_BOTH,
    KernelTypeTrans::KERNEL_HALF>;

#endif // HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_TCC_
