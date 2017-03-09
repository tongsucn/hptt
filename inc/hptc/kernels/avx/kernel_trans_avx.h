#pragma once
#ifndef HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_H_
#define HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_H_

#include <type_traits>

#include <hptc/types.h>
#include <hptc/config/config_trans.h>


namespace hptc {

#define REG_SIZE_BYTE_AVX 32


template <typename FloatType,
          KernelTypeTrans TYPE>
struct RegTypeDeducer<FloatType, TYPE,
    std::enable_if_t<(std::is_same<FloatType, float>::value or
            std::is_same<FloatType, FloatComplex>::value) and
        (TYPE == KernelTypeTrans::KERNEL_FULL or
            TYPE == KernelTypeTrans::KERNEL_LINE)>> {
  using type = __m256;
};

template <typename FloatType,
          KernelTypeTrans TYPE>
struct RegTypeDeducer<FloatType, TYPE,
    std::enable_if_t<(std::is_same<FloatType, double>::value or
            std::is_same<FloatType, DoubleComplex>::value) and
        (TYPE == KernelTypeTrans::KERNEL_FULL or
            TYPE == KernelTypeTrans::KERNEL_LINE)>> {
  using type = __m256d;
};

template <typename FloatType,
          KernelTypeTrans TYPE>
struct RegTypeDeducer<FloatType, TYPE,
    std::enable_if_t<(std::is_same<FloatType, float>::value or
            std::is_same<FloatType, FloatComplex>::value) and
        TYPE == KernelTypeTrans::KERNEL_HALF>> {
  using type = __m128;
};

template <typename FloatType,
          KernelTypeTrans TYPE>
struct RegTypeDeducer<FloatType, TYPE,
    std::enable_if_t<std::is_same<FloatType, double>::value and
        TYPE == KernelTypeTrans::KERNEL_HALF>> {
  using type = __m128d;
};

template <typename FloatType,
          KernelTypeTrans TYPE>
struct RegTypeDeducer<FloatType, TYPE,
    std::enable_if_t<std::is_same<FloatType, DoubleComplex>::value and
            TYPE == KernelTypeTrans::KERNEL_HALF>> {
  using type = double;
};


template <typename FloatType,
          KernelTypeTrans TYPE>
struct KernelTransAvxBase {
  using RegType = DeducedRegType<FloatType, TYPE>;
  using Deduced = DeducedFloatType<FloatType>;

  KernelTransAvxBase(Deduced coef_alpha, Deduced coef_beta);

  GenNumType get_kernel_width();
  GenNumType get_reg_num();

  template <KernelTypeTrans KERNEL = TYPE,
            std::enable_if_t<KERNEL == KernelTypeTrans::KERNEL_FULL or
                KERNEL == KernelTypeTrans::KERNEL_LINE> * = nullptr>
  DeducedRegType<float, KERNEL> reg_coef(float coef);

  template <KernelTypeTrans KERNEL = TYPE,
            std::enable_if_t<KERNEL == KernelTypeTrans::KERNEL_FULL or
                KERNEL == KernelTypeTrans::KERNEL_LINE> * = nullptr>
  DeducedRegType<double, KERNEL> reg_coef(double coef);

  template <KernelTypeTrans KERNEL = TYPE,
            std::enable_if_t<KERNEL == KernelTypeTrans::KERNEL_HALF> *
                = nullptr>
  DeducedRegType<float, KERNEL> reg_coef(float coef);

  template <KernelTypeTrans KERNEL = TYPE,
            std::enable_if_t<std::is_same<FloatType, double>::value and
                KERNEL == KernelTypeTrans::KERNEL_HALF> * = nullptr>
  DeducedRegType<double, KERNEL> reg_coef(double coef);

  template <KernelTypeTrans KERNEL = TYPE,
            std::enable_if_t<std::is_same<FloatType, DoubleComplex>::value and
                KERNEL == KernelTypeTrans::KERNEL_HALF> * = nullptr>
  DeducedRegType<DoubleComplex, KERNEL> reg_coef(double coef);

  RegType reg_alpha, reg_beta;
};


template <typename FloatType,
          CoefUsageTrans USAGE,
          KernelTypeTrans TYPE>
struct KernelTransAvx final : public KernelTransAvxBase<FloatType, TYPE> {
  using Deduced = DeducedFloatType<FloatType>;

  KernelTransAvx(Deduced coef_alpha, Deduced coef_beta);

  void operator()(const FloatType * RESTRICT input_data,
      FloatType * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride);
};


/*
 * Specialization for class KernelTransAvx
 */
template <CoefUsageTrans USAGE>
struct KernelTransAvx<float, USAGE, KernelTypeTrans::KERNEL_FULL> final
    : public KernelTransAvxBase<float, KernelTypeTrans::KERNEL_FULL> {
  KernelTransAvx(float coef_alpha, float coef_beta);
  void operator()(const float * RESTRICT input_data,
      float * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride);
};

template <CoefUsageTrans USAGE>
struct KernelTransAvx<double, USAGE, KernelTypeTrans::KERNEL_FULL> final
    : public KernelTransAvxBase<double, KernelTypeTrans::KERNEL_FULL> {
  KernelTransAvx(double coef_alpha, double coef_beta);
  void operator()(const double * RESTRICT input_data,
      double * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride);
};

template <CoefUsageTrans USAGE>
struct KernelTransAvx<FloatComplex, USAGE, KernelTypeTrans::KERNEL_FULL> final
    : public KernelTransAvxBase<FloatComplex, KernelTypeTrans::KERNEL_FULL> {
  KernelTransAvx(float coef_alpha, float coef_beta);
  void operator()(const FloatComplex * RESTRICT input_data,
      FloatComplex * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride);
};

template <CoefUsageTrans USAGE>
struct KernelTransAvx<DoubleComplex, USAGE, KernelTypeTrans::KERNEL_FULL> final
    : public KernelTransAvxBase<DoubleComplex, KernelTypeTrans::KERNEL_FULL> {
  KernelTransAvx(double coef_alpha, double coef_beta);
  void operator()(const DoubleComplex * RESTRICT input_data,
      DoubleComplex * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride);
};

template <CoefUsageTrans USAGE>
struct KernelTransAvx<float, USAGE, KernelTypeTrans::KERNEL_HALF> final
    : public KernelTransAvxBase<float, KernelTypeTrans::KERNEL_HALF> {
  KernelTransAvx(float coef_alpha, float coef_beta);
  void operator()(const float * RESTRICT input_data,
      float * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride);
};

template <CoefUsageTrans USAGE>
struct KernelTransAvx<double, USAGE, KernelTypeTrans::KERNEL_HALF> final
    : public KernelTransAvxBase<double, KernelTypeTrans::KERNEL_HALF> {
  KernelTransAvx(double coef_alpha, double coef_beta);
  void operator()(const double * RESTRICT input_data,
      double * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride);
};

template <CoefUsageTrans USAGE>
struct KernelTransAvx<FloatComplex, USAGE, KernelTypeTrans::KERNEL_HALF> final
    : public KernelTransAvxBase<FloatComplex, KernelTypeTrans::KERNEL_HALF> {
  KernelTransAvx(float coef_alpha, float coef_beta);
  void operator()(const FloatComplex * RESTRICT input_data,
      FloatComplex * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride);
};

template <CoefUsageTrans USAGE>
struct KernelTransAvx<DoubleComplex, USAGE, KernelTypeTrans::KERNEL_HALF> final
    : public KernelTransAvxBase<DoubleComplex, KernelTypeTrans::KERNEL_HALF> {
  KernelTransAvx(double coef_alpha, double coef_beta);
  void operator()(const DoubleComplex * RESTRICT input_data,
      DoubleComplex * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride);
};

template <CoefUsageTrans USAGE>
struct KernelTransAvx<float, USAGE, KernelTypeTrans::KERNEL_LINE> final
    : public KernelTransAvxBase<float, KernelTypeTrans::KERNEL_LINE> {
  KernelTransAvx(float coef_alpha, float coef_beta);
  void operator()(const float * RESTRICT input_data,
      float * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride);
};

template <CoefUsageTrans USAGE>
struct KernelTransAvx<double, USAGE, KernelTypeTrans::KERNEL_LINE> final
    : public KernelTransAvxBase<double, KernelTypeTrans::KERNEL_LINE> {
  KernelTransAvx(double coef_alpha, double coef_beta);
  void operator()(const double * RESTRICT input_data,
      double * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride);
};

template <CoefUsageTrans USAGE>
struct KernelTransAvx<FloatComplex, USAGE, KernelTypeTrans::KERNEL_LINE> final
    : public KernelTransAvxBase<FloatComplex, KernelTypeTrans::KERNEL_LINE> {
  KernelTransAvx(float coef_alpha, float coef_beta);
  void operator()(const FloatComplex * RESTRICT input_data,
      FloatComplex * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride);
};

template <CoefUsageTrans USAGE>
struct KernelTransAvx<DoubleComplex, USAGE, KernelTypeTrans::KERNEL_LINE> final
    : public KernelTransAvxBase<DoubleComplex, KernelTypeTrans::KERNEL_LINE> {
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

}

#endif // HPTC_KERNELS_AVX_KERNEL_TRANS_AVX_H_
