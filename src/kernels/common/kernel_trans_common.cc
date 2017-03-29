#include <hptc/kernels/common/kernel_trans_common.h>

#include <hptc/types.h>
#include <hptc/compat.h>


void set_trans_coef_full_s_(void *reg, const float coef) {
  *reinterpret_cast<float *>(reg) = coef;
}


void set_trans_coef_full_d_(void *reg, const double coef) {
  *reinterpret_cast<double *>(reg) = coef;
}


void set_trans_coef_half_s_(void *reg, const float coef) {
  *reinterpret_cast<float *>(reg) = coef;
}


void set_trans_coef_half_d_(void *reg, const double coef) {
  *reinterpret_cast<double *>(reg) = coef;
}


template <typename FloatType>
HPTC_INL static void trans_impl_full_(const FloatType * RESTRICT input_data,
    FloatType * RESTRICT output_data, const hptc::TensorIdx input_stride,
    const hptc::TensorIdx output_stride,
    const hptc::DeducedFloatType<FloatType> reg_alpha,
    const hptc::DeducedFloatType<FloatType> reg_beta) {
  // Get number of elements to be processed in on row
  constexpr auto KN_WIDTH = REG_SIZE / sizeof(FloatType);

  for (hptc::TensorUInt ncont_idx = 0; ncont_idx < KN_WIDTH; ++ncont_idx) {
    for (hptc::TensorUInt cont_idx = 0; cont_idx < KN_WIDTH; ++cont_idx) {
      const auto input_idx = cont_idx + ncont_idx * input_stride,
            output_idx = ncont_idx + cont_idx * output_stride;
      output_data[output_idx] = reg_alpha * input_data[input_idx]
            + reg_beta * output_data[output_idx];
    }
  }
}


void exec_trans_full_s_(const float *input_data, float *output_data,
    const hptc::TensorIdx input_stride, const hptc::TensorIdx output_stride,
    const void *alpha, const void *beta) {
  auto reg_alpha = *reinterpret_cast<const float *>(alpha);
  auto reg_beta = *reinterpret_cast<const float *>(beta);

  trans_impl_full_<float>(input_data, output_data, input_stride, output_stride,
      reg_alpha, reg_beta);
}


void exec_trans_full_d_(const double *input_data, double *output_data,
    const hptc::TensorIdx input_stride, const hptc::TensorIdx output_stride,
    const void *alpha, const void *beta) {
  auto reg_alpha = *reinterpret_cast<const double *>(alpha);
  auto reg_beta = *reinterpret_cast<const double *>(beta);

  trans_impl_full_<double>(input_data, output_data, input_stride, output_stride,
      reg_alpha, reg_beta);
}


void exec_trans_full_c_(const hptc::FloatComplex *input_data,
    hptc::FloatComplex *output_data, const hptc::TensorIdx input_stride,
    const hptc::TensorIdx output_stride, const void *alpha,
    const void *beta) {
  auto reg_alpha = *reinterpret_cast<const float *>(alpha);
  auto reg_beta = *reinterpret_cast<const float *>(beta);

  trans_impl_full_<hptc::FloatComplex>(input_data, output_data, input_stride,
      output_stride, reg_alpha, reg_beta);
}


void exec_trans_full_z_(const hptc::DoubleComplex *input_data,
    hptc::DoubleComplex *output_data, const hptc::TensorIdx input_stride,
    const hptc::TensorIdx output_stride, const void *alpha,
    const void *beta) {
  auto reg_alpha = *reinterpret_cast<const double *>(alpha);
  auto reg_beta = *reinterpret_cast<const double *>(beta);

  trans_impl_full_<hptc::DoubleComplex>(input_data, output_data, input_stride,
      output_stride, reg_alpha, reg_beta);
}


template <typename FloatType>
HPTC_INL static void trans_impl_half_(const FloatType * RESTRICT input_data,
    FloatType * RESTRICT output_data, const hptc::TensorIdx input_stride,
    const hptc::TensorIdx output_stride,
    const hptc::DeducedFloatType<FloatType> reg_alpha,
    const hptc::DeducedFloatType<FloatType> reg_beta) {
  // Get number of elements to be processed in on row
  constexpr auto KN_WIDTH = REG_SIZE / sizeof(FloatType) / 2;

  for (hptc::TensorUInt ncont_idx = 0; ncont_idx < KN_WIDTH; ++ncont_idx) {
    for (hptc::TensorUInt cont_idx = 0; cont_idx < KN_WIDTH; ++cont_idx) {
      const auto input_idx = cont_idx + ncont_idx * input_stride,
            output_idx = ncont_idx + cont_idx * output_stride;
      output_data[output_idx] = reg_alpha * input_data[input_idx]
            + reg_beta * output_data[output_idx];
    }
  }
}


void exec_trans_half_s_(const float *input_data, float *output_data,
    const hptc::TensorIdx input_stride, const hptc::TensorIdx output_stride,
    const void *alpha, const void *beta) {
  auto reg_alpha = *reinterpret_cast<const float *>(alpha);
  auto reg_beta = *reinterpret_cast<const float *>(beta);

  trans_impl_half_<float>(input_data, output_data, input_stride, output_stride,
      reg_alpha, reg_beta);
}


void exec_trans_half_d_(const double *input_data, double *output_data,
    const hptc::TensorIdx input_stride, const hptc::TensorIdx output_stride,
    const void *alpha, const void *beta) {
  auto reg_alpha = *reinterpret_cast<const double *>(alpha);
  auto reg_beta = *reinterpret_cast<const double *>(beta);

  trans_impl_half_<double>(input_data, output_data, input_stride, output_stride,
      reg_alpha, reg_beta);
}


void exec_trans_half_c_(const hptc::FloatComplex *input_data,
    hptc::FloatComplex *output_data, const hptc::TensorIdx input_stride,
    const hptc::TensorIdx output_stride, const void *alpha,
    const void *beta) {
  auto reg_alpha = *reinterpret_cast<const float *>(alpha);
  auto reg_beta = *reinterpret_cast<const float *>(beta);

  trans_impl_half_<hptc::FloatComplex>(input_data, output_data, input_stride,
      output_stride, reg_alpha, reg_beta);
}


void exec_trans_half_z_(const hptc::DoubleComplex *input_data,
    hptc::DoubleComplex *output_data, const hptc::TensorIdx input_stride,
    const hptc::TensorIdx output_stride, const void *alpha,
    const void *beta) {
  auto reg_alpha = *reinterpret_cast<const double *>(alpha);
  auto reg_beta = *reinterpret_cast<const double *>(beta);

  trans_impl_half_<hptc::DoubleComplex>(input_data, output_data, input_stride,
      output_stride, reg_alpha, reg_beta);
}
