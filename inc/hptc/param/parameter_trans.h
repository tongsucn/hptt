#pragma once
#ifndef HPTC_PARAM_PARAMETER_TRANS_H_
#define HPTC_PARAM_PARAMETER_TRANS_H_

#include <array>
#include <algorithm>

#include <hptc/types.h>
#include <hptc/tensor.h>


namespace hptc {

enum class CoefUsage : GenNumType {
  USE_NONE  = 0x0,
  USE_ALPHA = 0x1,
  USE_BETA  = 0x2,
  USE_BOTH  = 0x3
};


template <typename FloatType,
          TensorOrder ORDER,
          MemLayout LAYOUT,
          CoefUsage USAGE>
struct ParamTrans {
  ParamTrans(const TensorWrapper<FloatType, ORDER, LAYOUT> &input_tensor,
      const TensorWrapper<FloatType, ORDER, LAYOUT> &output_tensor,
      const std::array<TensorOrder, ORDER> &perm,
      DeducedFloatType<FloatType> alpha = 1,
      DeducedFloatType<FloatType> beta = 0);

  TensorWrapper<FloatType, ORDER, LAYOUT> input_tensor, output_tensor;
  DeducedFloatType<FloatType> alpha, beta;

  constexpr static CoefUsage COEF_USAGE = USAGE;
  TensorOrder perm[ORDER];
  TensorIdx input_stride, output_stride;
};


/*
 * Implementation for class ParamTrans
 */
template <typename FloatType,
          TensorOrder ORDER,
          MemLayout LAYOUT,
          CoefUsage USAGE>
ParamTrans<FloatType, ORDER, LAYOUT, USAGE>::ParamTrans(
    const TensorWrapper<FloatType, ORDER, LAYOUT> &input_tensor,
    const TensorWrapper<FloatType, ORDER, LAYOUT> &output_tensor,
    const std::array<TensorOrder, ORDER> &perm,
    DeducedFloatType<FloatType> alpha, DeducedFloatType<FloatType> beta)
    : input_tensor(input_tensor), output_tensor(output_tensor),
      alpha(alpha), beta(beta),
      input_stride(1), output_stride(1) {
  // Initialize permutation map
  copy(perm.begin(), perm.end(), this->perm);

  // Initialize access stride
  if (MemLayout::COL_MAJOR == LAYOUT) {
    for (TensorIdx idx = 0; idx < this->perm[0]; ++idx)
      this->input_stride *= input_tensor.get_outer_size()[idx];
    for (TensorIdx idx = 0; 0 != this->perm[idx]; ++idx)
      this->output_stride *= output_tensor.get_outer_size()[idx];
  }
  else {
    for (TensorIdx idx = ORDER - 1; idx > this->perm[ORDER - 1]; --idx)
      this->input_stride *= input_tensor.get_outer_size()[idx];
    for (TensorIdx idx = ORDER - 1; ORDER - 1 != this->perm[idx]; --idx)
      this->output_stride *= output_stride.get_outer_size()[idx];
  }
}

}

#endif // HPTC_PARAM_PARAMETER_TRANS_H_
