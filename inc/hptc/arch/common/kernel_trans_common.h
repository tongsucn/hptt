#pragma once
#ifndef HPTC_ARCH_COMMON_KERNEL_TRANS_COMMON_H_
#define HPTC_ARCH_COMMON_KERNEL_TRANS_COMMON_H_

#include <hptc/types.h>


extern "C" {

extern constexpr hptc::TensorUInt REG_SIZE = 32;


void set_trans_coef_full_s_(void *reg, const float coef);
void set_trans_coef_half_s_(void *reg, const float coef);
void set_trans_coef_linear_s_(void *reg, const float coef);
void set_trans_coef_full_d_(void *reg, const double coef);
void set_trans_coef_half_d_(void *reg, const double coef);
void set_trans_coef_linear_d_(void *reg, const double coef);


void exec_trans_full_s_(const float *input_data, float *output_data,
    const hptc::TensorIdx input_stride, const hptc::TensorIdx output_stride,
    const void *alpha, const void *beta);
void exec_trans_full_d_(const double *input_data, double *output_data,
    const hptc::TensorIdx input_stride, const hptc::TensorIdx output_stride,
    const void *alpha, const void *beta);
void exec_trans_full_c_(const hptc::FloatComplex *input_data,
    hptc::FloatComplex *output_data, const hptc::TensorIdx input_stride,
    const hptc::TensorIdx output_stride, const void *alpha, const void *beta);
void exec_trans_full_z_(const hptc::DoubleComplex *input_data,
    hptc::DoubleComplex *output_data, const hptc::TensorIdx input_stride,
    const hptc::TensorIdx output_stride, const void *alpha, const void *beta);

void exec_trans_half_s_(const float *input_data, float *output_data,
    const hptc::TensorIdx input_stride, const hptc::TensorIdx output_stride,
    const void *alpha, const void *beta);
void exec_trans_half_d_(const double *input_data, double *output_data,
    const hptc::TensorIdx input_stride, const hptc::TensorIdx output_stride,
    const void *alpha, const void *beta);
void exec_trans_half_c_(const hptc::FloatComplex *input_data,
    hptc::FloatComplex *output_data, const hptc::TensorIdx input_stride,
    const hptc::TensorIdx output_stride, const void *alpha, const void *beta);
void exec_trans_half_z_(const hptc::DoubleComplex *input_data,
    hptc::DoubleComplex *output_data, const hptc::TensorIdx input_stride,
    const hptc::TensorIdx output_stride, const void *alpha, const void *beta);

void exec_trans_linear_s_(const float *input_data, float *output_data,
    const hptc::TensorIdx input_stride, const hptc::TensorIdx output_stride,
    const void *alpha, const void *beta);
void exec_trans_linear_d_(const double *input_data, double *output_data,
    const hptc::TensorIdx input_stride, const hptc::TensorIdx output_stride,
    const void *alpha, const void *beta);
void exec_trans_linear_c_(const hptc::FloatComplex *input_data,
    hptc::FloatComplex *output_data, const hptc::TensorIdx input_stride,
    const hptc::TensorIdx output_stride, const void *alpha, const void *beta);
void exec_trans_linear_z_(const hptc::DoubleComplex *input_data,
    hptc::DoubleComplex *output_data, const hptc::TensorIdx input_stride,
    const hptc::TensorIdx output_stride, const void *alpha, const void *beta);

}

#endif // HPTC_ARCH_COMMON_KERNEL_TRANS_COMMON_H_
