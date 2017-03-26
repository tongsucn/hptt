#pragma once
#ifndef HPTC_KERNELS_AVX2_KERNEL_TRANS_AVX2_TCC_
#define HPTC_KERNELS_AVX2_KERNEL_TRANS_AVX2_TCC_

/*
 * Partial specialization for class KernelTransAvx2
 */
template <CoefUsageTrans USAGE>
struct KernelTransAvx2<float, USAGE, KernelTypeTrans::KERNEL_FULL> final
    : public KernelTransAvx2Base<float, KernelTypeTrans::KERNEL_FULL> {
  static RegType reg_coef(const DeducedFloatType<float> coef);

  void operator()(const float * RESTRICT input_data,
      float * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride, const RegType &reg_alpha,
      const RegType &reg_beta) const;
};


template <CoefUsageTrans USAGE>
struct KernelTransAvx2<double, USAGE, KernelTypeTrans::KERNEL_FULL> final
    : public KernelTransAvx2Base<double, KernelTypeTrans::KERNEL_FULL> {
  static RegType reg_coef(const DeducedFloatType<double> coef);

  void operator()(const double * RESTRICT input_data,
      double * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride, const RegType &reg_alpha,
      const RegType &reg_beta) const;
};


template <CoefUsageTrans USAGE>
struct KernelTransAvx2<FloatComplex, USAGE, KernelTypeTrans::KERNEL_FULL> final
    : public KernelTransAvx2Base<FloatComplex, KernelTypeTrans::KERNEL_FULL> {
  static RegType reg_coef(const DeducedFloatType<FloatComplex> coef);

  void operator()(const FloatComplex * RESTRICT input_data,
      FloatComplex * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride, const RegType &reg_alpha,
      const RegType &reg_beta) const;
};


template <CoefUsageTrans USAGE>
struct KernelTransAvx2<DoubleComplex, USAGE, KernelTypeTrans::KERNEL_FULL> final
    : public KernelTransAvx2Base<DoubleComplex, KernelTypeTrans::KERNEL_FULL> {
  static RegType reg_coef(const DeducedFloatType<DoubleComplex> coef);

  void operator()(const DoubleComplex * RESTRICT input_data,
      DoubleComplex * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride, const RegType &reg_alpha,
      const RegType &reg_beta) const;
};


template <CoefUsageTrans USAGE>
struct KernelTransAvx2<float, USAGE, KernelTypeTrans::KERNEL_HALF> final
    : public KernelTransAvx2Base<float, KernelTypeTrans::KERNEL_HALF> {
  static RegType reg_coef(const DeducedFloatType<float> coef);

  void operator()(const float * RESTRICT input_data,
      float * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride, const RegType &reg_alpha,
      const RegType &reg_beta) const;
};


template <CoefUsageTrans USAGE>
struct KernelTransAvx2<double, USAGE, KernelTypeTrans::KERNEL_HALF> final
    : public KernelTransAvx2Base<double, KernelTypeTrans::KERNEL_HALF> {
  static RegType reg_coef(const DeducedFloatType<double> coef);

  void operator()(const double * RESTRICT input_data,
      double * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride, const RegType &reg_alpha,
      const RegType &reg_beta) const;
};


template <CoefUsageTrans USAGE>
struct KernelTransAvx2<FloatComplex, USAGE, KernelTypeTrans::KERNEL_HALF> final
    : public KernelTransAvx2Base<FloatComplex, KernelTypeTrans::KERNEL_HALF> {
  static RegType reg_coef(const DeducedFloatType<FloatComplex> coef);

  void operator()(const FloatComplex * RESTRICT input_data,
      FloatComplex * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride, const RegType &reg_alpha,
      const RegType &reg_beta) const;
};


template <CoefUsageTrans USAGE>
struct KernelTransAvx2<DoubleComplex, USAGE, KernelTypeTrans::KERNEL_HALF> final
    : public KernelTransAvx2Base<DoubleComplex, KernelTypeTrans::KERNEL_HALF> {
  static RegType reg_coef(const DeducedFloatType<DoubleComplex> coef);

  void operator()(const DoubleComplex * RESTRICT input_data,
      DoubleComplex * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride, const RegType &reg_alpha,
      const RegType &reg_beta) const;
};


/*
 * Explicit instantiation declaration for struct KernelTransAvx2
 */
extern template struct KernelTransAvx2<float, CoefUsageTrans::USE_NONE,
         KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvx2<float, CoefUsageTrans::USE_NONE,
         KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvx2<float, CoefUsageTrans::USE_ALPHA,
         KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvx2<float, CoefUsageTrans::USE_ALPHA,
         KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvx2<float, CoefUsageTrans::USE_BETA,
         KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvx2<float, CoefUsageTrans::USE_BETA,
         KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvx2<float, CoefUsageTrans::USE_BOTH,
         KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvx2<float, CoefUsageTrans::USE_BOTH,
         KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvx2<double, CoefUsageTrans::USE_NONE,
         KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvx2<double, CoefUsageTrans::USE_NONE,
         KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvx2<double, CoefUsageTrans::USE_ALPHA,
         KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvx2<double, CoefUsageTrans::USE_ALPHA,
         KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvx2<double, CoefUsageTrans::USE_BETA,
         KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvx2<double, CoefUsageTrans::USE_BETA,
         KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvx2<double, CoefUsageTrans::USE_BOTH,
         KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvx2<double, CoefUsageTrans::USE_BOTH,
         KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvx2<FloatComplex, CoefUsageTrans::USE_NONE,
         KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvx2<FloatComplex, CoefUsageTrans::USE_NONE,
         KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvx2<FloatComplex, CoefUsageTrans::USE_ALPHA,
         KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvx2<FloatComplex, CoefUsageTrans::USE_ALPHA,
         KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvx2<FloatComplex, CoefUsageTrans::USE_BETA,
         KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvx2<FloatComplex, CoefUsageTrans::USE_BETA,
         KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvx2<FloatComplex, CoefUsageTrans::USE_BOTH,
         KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvx2<FloatComplex, CoefUsageTrans::USE_BOTH,
         KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvx2<DoubleComplex, CoefUsageTrans::USE_NONE,
         KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvx2<DoubleComplex, CoefUsageTrans::USE_NONE,
         KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvx2<DoubleComplex, CoefUsageTrans::USE_ALPHA,
         KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvx2<DoubleComplex, CoefUsageTrans::USE_ALPHA,
         KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvx2<DoubleComplex, CoefUsageTrans::USE_BETA,
         KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvx2<DoubleComplex, CoefUsageTrans::USE_BETA,
         KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvx2<DoubleComplex, CoefUsageTrans::USE_BOTH,
         KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvx2<DoubleComplex, CoefUsageTrans::USE_BOTH,
         KernelTypeTrans::KERNEL_HALF>;

#endif // HPTC_KERNELS_AVX2_KERNEL_TRANS_AVX2_TCC_
