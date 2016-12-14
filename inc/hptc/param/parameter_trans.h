#pragma once
#ifndef HPTC_PARAM_PARAMETER_TRANS_H_
#define HPTC_PARAM_PARAMETER_TRANS_H_

#include <memory>
#include <vector>
#include <algorithm>

#include <hptc/types.h>
#include <hptc/tensor.h>


namespace hptc {

template <typename FloatType>
struct ParamTrans {
  using Deduced = DeducedFloatType<FloatType>;

  ParamTrans(const TensorWrapper<FloatType> &input_tensor,
      const TensorWrapper<FloatType> &output_tensor,
      const std::vector<TensorDim> &perm,
      Deduced alpha = 1.0, Deduced beta = 0.0);

  ~ParamTrans();


  TensorWrapper<FloatType> input_tensor, output_tensor;
  TensorDim *perm;
  Deduced alpha, beta;

  TensorIdx input_stride, output_stride;
  TensorIdx *macro_loop_idx, **macro_loop_perm_idx;

private:
  const TensorDim tensor_dim_;
};


/*
 * Implementation for class ParamTrans
 */
template <typename FloatType>
ParamTrans<FloatType>::ParamTrans(const TensorWrapper<FloatType> &input_tensor,
    const TensorWrapper<FloatType> &output_tensor,
    const std::vector<TensorDim> &perm, Deduced alpha, Deduced beta)
    : input_tensor(input_tensor),
      output_tensor(output_tensor),
      perm(new TensorDim [perm.size()]), alpha(alpha), beta(beta),
      input_stride(1), output_stride(1),
      tensor_dim_(input_tensor.get_size().get_dim()) {
  // Initialize permutation map
  copy(perm.begin(), perm.end(), this->perm);

  // Initialize access stride for input tensor
  TensorIdx upper = this->tensor_dim_ - this->perm[0] - 1;
  for (TensorIdx idx = this->tensor_dim_ - 1; idx > upper; --idx)
    this->input_stride *= input_tensor.get_outer_size()[idx];

  // Initialize access stride for output tensor
  for (TensorIdx idx = 0; 0 != this->perm[idx]; ++idx)
    this->output_stride
        *= output_tensor.get_outer_size()[this->tensor_dim_ - 1 - idx];

  // Initialize loop indices
  this->macro_loop_idx = new TensorIdx [this->tensor_dim_];
  this->macro_loop_perm_idx = new TensorIdx * [this->tensor_dim_];
  for (TensorDim idx = 0; idx < this->tensor_dim_; ++idx)
    this->macro_loop_perm_idx[idx] = &this->macro_loop_idx[this->perm[idx]];
}


template <typename FloatType>
ParamTrans<FloatType>::~ParamTrans() {
  delete [] this->perm;
  delete [] this->macro_loop_idx;
}

}

#endif // HPTC_PARAM_PARAMETER_TRANS_H_
