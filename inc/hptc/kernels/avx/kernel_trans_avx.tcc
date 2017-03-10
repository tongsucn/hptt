#pragma once
#ifndef HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_TCC_
#define HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_TCC_

/*
 * Specialization for class KernelTransAvx
 */
template <CoefUsageTrans USAGE>
struct KernelTransAvx<float, USAGE, KernelTypeTrans::KERNEL_FULL> final
    : public KernelTransAvxBase<float, KernelTypeTrans::KERNEL_FULL> {
  using FLOAT = float;
  KernelTransAvx(float coef_alpha, float coef_beta);
  void operator()(const float * RESTRICT input_data,
      float * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride);
};

template <CoefUsageTrans USAGE>
struct KernelTransAvx<double, USAGE, KernelTypeTrans::KERNEL_FULL> final
    : public KernelTransAvxBase<double, KernelTypeTrans::KERNEL_FULL> {
  using FLOAT = double;
  KernelTransAvx(double coef_alpha, double coef_beta);
  void operator()(const double * RESTRICT input_data,
      double * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride);
};

template <CoefUsageTrans USAGE>
struct KernelTransAvx<FloatComplex, USAGE, KernelTypeTrans::KERNEL_FULL> final
    : public KernelTransAvxBase<FloatComplex, KernelTypeTrans::KERNEL_FULL> {
  using FLOAT = FloatComplex;
  KernelTransAvx(float coef_alpha, float coef_beta);
  void operator()(const FloatComplex * RESTRICT input_data,
      FloatComplex * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride);
};

template <CoefUsageTrans USAGE>
struct KernelTransAvx<DoubleComplex, USAGE, KernelTypeTrans::KERNEL_FULL> final
    : public KernelTransAvxBase<DoubleComplex, KernelTypeTrans::KERNEL_FULL> {
  using FLOAT = DoubleComplex;
  KernelTransAvx(double coef_alpha, double coef_beta);
  void operator()(const DoubleComplex * RESTRICT input_data,
      DoubleComplex * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride);
};

template <CoefUsageTrans USAGE>
struct KernelTransAvx<float, USAGE, KernelTypeTrans::KERNEL_HALF> final
    : public KernelTransAvxBase<float, KernelTypeTrans::KERNEL_HALF> {
  using FLOAT = float;
  KernelTransAvx(float coef_alpha, float coef_beta);
  void operator()(const float * RESTRICT input_data,
      float * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride);
};

template <CoefUsageTrans USAGE>
struct KernelTransAvx<double, USAGE, KernelTypeTrans::KERNEL_HALF> final
    : public KernelTransAvxBase<double, KernelTypeTrans::KERNEL_HALF> {
  using FLOAT = double;
  KernelTransAvx(double coef_alpha, double coef_beta);
  void operator()(const double * RESTRICT input_data,
      double * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride);
};

template <CoefUsageTrans USAGE>
struct KernelTransAvx<FloatComplex, USAGE, KernelTypeTrans::KERNEL_HALF> final
    : public KernelTransAvxBase<FloatComplex, KernelTypeTrans::KERNEL_HALF> {
  using FLOAT = FloatComplex;
  KernelTransAvx(float coef_alpha, float coef_beta);
  void operator()(const FloatComplex * RESTRICT input_data,
      FloatComplex * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride);
};

template <CoefUsageTrans USAGE>
struct KernelTransAvx<DoubleComplex, USAGE, KernelTypeTrans::KERNEL_HALF> final
    : public KernelTransAvxBase<DoubleComplex, KernelTypeTrans::KERNEL_HALF> {
  using FLOAT = DoubleComplex;
  KernelTransAvx(double coef_alpha, double coef_beta);
  void operator()(const DoubleComplex * RESTRICT input_data,
      DoubleComplex * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride);
};

template <CoefUsageTrans USAGE>
struct KernelTransAvx<float, USAGE, KernelTypeTrans::KERNEL_LINE> final
    : public KernelTransAvxBase<float, KernelTypeTrans::KERNEL_LINE> {
  using FLOAT = float;
  KernelTransAvx(float coef_alpha, float coef_beta);
  void operator()(const float * RESTRICT input_data,
      float * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride);
};

template <CoefUsageTrans USAGE>
struct KernelTransAvx<double, USAGE, KernelTypeTrans::KERNEL_LINE> final
    : public KernelTransAvxBase<double, KernelTypeTrans::KERNEL_LINE> {
  using FLOAT = double;
  KernelTransAvx(double coef_alpha, double coef_beta);
  void operator()(const double * RESTRICT input_data,
      double * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride);
};

template <CoefUsageTrans USAGE>
struct KernelTransAvx<FloatComplex, USAGE, KernelTypeTrans::KERNEL_LINE> final
    : public KernelTransAvxBase<FloatComplex, KernelTypeTrans::KERNEL_LINE> {
  using FLOAT = FloatComplex;
  KernelTransAvx(float coef_alpha, float coef_beta);
  void operator()(const FloatComplex * RESTRICT input_data,
      FloatComplex * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride);
};

template <CoefUsageTrans USAGE>
struct KernelTransAvx<DoubleComplex, USAGE, KernelTypeTrans::KERNEL_LINE> final
    : public KernelTransAvxBase<DoubleComplex, KernelTypeTrans::KERNEL_LINE> {
  using FLOAT = DoubleComplex;
  KernelTransAvx(double coef_alpha, double coef_beta);
  void operator()(const DoubleComplex * RESTRICT input_data,
      DoubleComplex * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride);
};


/*
 * Avoid template instantiation for struct KernelTransAvxBase
 */
extern template struct KernelTransAvxBase<float, KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvxBase<float, KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvxBase<float, KernelTypeTrans::KERNEL_LINE>;
extern template struct KernelTransAvxBase<double, KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvxBase<double, KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvxBase<double, KernelTypeTrans::KERNEL_LINE>;
extern template struct KernelTransAvxBase<FloatComplex,
       KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvxBase<FloatComplex,
       KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvxBase<FloatComplex,
       KernelTypeTrans::KERNEL_LINE>;
extern template struct KernelTransAvxBase<DoubleComplex,
       KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvxBase<DoubleComplex,
       KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvxBase<DoubleComplex,
       KernelTypeTrans::KERNEL_LINE>;


/*
 * Avoid template instantiation for struct KernelTransAvx
 */
extern template struct KernelTransAvx<float, CoefUsageTrans::USE_NONE,
         KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvx<float, CoefUsageTrans::USE_NONE,
         KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvx<float, CoefUsageTrans::USE_NONE,
         KernelTypeTrans::KERNEL_LINE>;
extern template struct KernelTransAvx<float, CoefUsageTrans::USE_ALPHA,
         KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvx<float, CoefUsageTrans::USE_ALPHA,
         KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvx<float, CoefUsageTrans::USE_ALPHA,
         KernelTypeTrans::KERNEL_LINE>;
extern template struct KernelTransAvx<float, CoefUsageTrans::USE_BETA,
         KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvx<float, CoefUsageTrans::USE_BETA,
         KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvx<float, CoefUsageTrans::USE_BETA,
         KernelTypeTrans::KERNEL_LINE>;
extern template struct KernelTransAvx<float, CoefUsageTrans::USE_BOTH,
         KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvx<float, CoefUsageTrans::USE_BOTH,
         KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvx<float, CoefUsageTrans::USE_BOTH,
         KernelTypeTrans::KERNEL_LINE>;
extern template struct KernelTransAvx<double, CoefUsageTrans::USE_NONE,
         KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvx<double, CoefUsageTrans::USE_NONE,
         KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvx<double, CoefUsageTrans::USE_NONE,
         KernelTypeTrans::KERNEL_LINE>;
extern template struct KernelTransAvx<double, CoefUsageTrans::USE_ALPHA,
         KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvx<double, CoefUsageTrans::USE_ALPHA,
         KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvx<double, CoefUsageTrans::USE_ALPHA,
         KernelTypeTrans::KERNEL_LINE>;
extern template struct KernelTransAvx<double, CoefUsageTrans::USE_BETA,
         KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvx<double, CoefUsageTrans::USE_BETA,
         KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvx<double, CoefUsageTrans::USE_BETA,
         KernelTypeTrans::KERNEL_LINE>;
extern template struct KernelTransAvx<double, CoefUsageTrans::USE_BOTH,
         KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvx<double, CoefUsageTrans::USE_BOTH,
         KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvx<double, CoefUsageTrans::USE_BOTH,
         KernelTypeTrans::KERNEL_LINE>;
extern template struct KernelTransAvx<FloatComplex, CoefUsageTrans::USE_NONE,
         KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvx<FloatComplex, CoefUsageTrans::USE_NONE,
         KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvx<FloatComplex, CoefUsageTrans::USE_NONE,
         KernelTypeTrans::KERNEL_LINE>;
extern template struct KernelTransAvx<FloatComplex, CoefUsageTrans::USE_ALPHA,
         KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvx<FloatComplex, CoefUsageTrans::USE_ALPHA,
         KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvx<FloatComplex, CoefUsageTrans::USE_ALPHA,
         KernelTypeTrans::KERNEL_LINE>;
extern template struct KernelTransAvx<FloatComplex, CoefUsageTrans::USE_BETA,
         KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvx<FloatComplex, CoefUsageTrans::USE_BETA,
         KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvx<FloatComplex, CoefUsageTrans::USE_BETA,
         KernelTypeTrans::KERNEL_LINE>;
extern template struct KernelTransAvx<FloatComplex, CoefUsageTrans::USE_BOTH,
         KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvx<FloatComplex, CoefUsageTrans::USE_BOTH,
         KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvx<FloatComplex, CoefUsageTrans::USE_BOTH,
         KernelTypeTrans::KERNEL_LINE>;
extern template struct KernelTransAvx<DoubleComplex, CoefUsageTrans::USE_NONE,
         KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvx<DoubleComplex, CoefUsageTrans::USE_NONE,
         KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvx<DoubleComplex, CoefUsageTrans::USE_NONE,
         KernelTypeTrans::KERNEL_LINE>;
extern template struct KernelTransAvx<DoubleComplex, CoefUsageTrans::USE_ALPHA,
         KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvx<DoubleComplex, CoefUsageTrans::USE_ALPHA,
         KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvx<DoubleComplex, CoefUsageTrans::USE_ALPHA,
         KernelTypeTrans::KERNEL_LINE>;
extern template struct KernelTransAvx<DoubleComplex, CoefUsageTrans::USE_BETA,
         KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvx<DoubleComplex, CoefUsageTrans::USE_BETA,
         KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvx<DoubleComplex, CoefUsageTrans::USE_BETA,
         KernelTypeTrans::KERNEL_LINE>;
extern template struct KernelTransAvx<DoubleComplex, CoefUsageTrans::USE_BOTH,
         KernelTypeTrans::KERNEL_FULL>;
extern template struct KernelTransAvx<DoubleComplex, CoefUsageTrans::USE_BOTH,
         KernelTypeTrans::KERNEL_HALF>;
extern template struct KernelTransAvx<DoubleComplex, CoefUsageTrans::USE_BOTH,
         KernelTypeTrans::KERNEL_LINE>;

#endif // HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_TCC_
