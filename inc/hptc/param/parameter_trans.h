#pragma once
#ifndef HPTC_PARAM_PARAMETER_TRANS_H_
#define HPTC_PARAM_PARAMETER_TRANS_H_

#include <memory>
#include <array>
#include <algorithm>

#include <hptc/types.h>
#include <hptc/tensor.h>


namespace hptc {

template <typename FloatType,
          TensorOrder ORDER,
          MemLayout LAYOUT = COL_MAJOR>
struct ParamTrans {
  ParamTrans(const TensorWrapper<FloatType, ORDER, LAYOUT> &input_tensor,
      const TensorWrapper<FloatType, LAYOUT> &output_tensor,
      const std::array<TensorOrder, ORDER> &perm,
      DeducedFloatType<FloatType> alpha = 1.0,
      DeducedFloatType<FloatType> beta = 0.0);

  ParamTrans(const ParamTrans &param) = delete;
  ParamTrans &operator=(const ParamTrans &param) = delete;

  TensorWrapper<FloatType, ORDER, LAYOUT> input_tensor, output_tensor;
  TensorOrder perm[ORDER];
  DeducedFloatType<FloatType> alpha, beta;

  TensorIdx input_stride, output_stride;
  TensorIdx macro_loop_idx[ORDER];
  TensorIdx *macro_loop_perm_idx[ORDER];
};


/*
 * Implementation for class ParamTrans
 */
template <typename FloatType,
          TensorOrder ORDER,
          MemLayout LAYOUT = COL_MAJOR>
ParamTrans<FloatType, ORDER, LAYOUT>::ParamTrans(
    const TensorWrapper<FloatType, LAYOUT> &input_tensor,
    const TensorWrapper<FloatType, LAYOUT> &output_tensor,
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

  // Initialize loop indices
  for (TensorOrder idx = 0; idx < ORDER; ++idx)
    this->macro_loop_perm_idx[idx] = &this->macro_loop_idx[this->perm[idx]];
}

}

#endif // HPTC_PARAM_PARAMETER_TRANS_H_
