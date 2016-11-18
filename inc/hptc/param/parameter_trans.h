#pragma once
#ifndef HPTC_PARAMETER_TRANS_H_
#define HPTC_PARAMETER_TRANS_H_

#include <memory>
#include <vector>
#include <algorithm>

#include <hptc/types.h>
#include <hptc/tensor.h>

namespace hptc {

template <typename FloatType>
struct ParamTrans {
  ParamTrans() = delete;
  ParamTrans(const TensorWrapper<FloatType> &input_tensor,
      TensorWrapper<FloatType> &output_tensor, CoefficientType<FloatType> alpha,
      CoefficientType<FloatType> beta, std::initializer_list<TensorDim> perm);

  ParamTrans(const ParamTrans &param_obj);
  ParamTrans(ParamTrans &&param_obj) noexcept;

  ParamTrans &operator=(const ParamTrans &param_obj) = delete;

  ~ParamTrans();

  const TensorWrapper<FloatType> &input_tensor;
  TensorWrapper<FloatType> &output_tensor;

  CoefficientType<FloatType> alpha;
  CoefficientType<FloatType> beta;

  TensorDim tensor_dim;
  TensorIdx input_offset;
  TensorIdx output_offset;
  std::vector<TensorDim> perm;
  TensorIdx *macro_loop_idx;
  TensorIdx *remainder_loop_idx;
};


/*
 * Implementation for class ParamTrans
 */
template <typename FloatType>
ParamTrans<FloatType>::ParamTrans(const TensorWrapper<FloatType> &input_tensor,
    TensorWrapper<FloatType> &output_tensor, CoefficientType<FloatType> alpha,
    CoefficientType<FloatType> beta, std::initializer_list<TensorDim> perm)
    : input_tensor(input_tensor),
      output_tensor(output_tensor),
      alpha(alpha),
      beta(beta),
      tensor_dim(input_tensor.get_size().get_dim()),
      input_offset(input_tensor.get_outer_size()[0]),
      output_offset(output_tensor.get_outer_size()[0]),
      perm(perm) {
  for (int dim_idx = 1; dim_idx < this->tensor_dim; ++dim_idx) {
    input_offset *= input_tensor.get_outer_size()[dim_idx];
    output_offset *= output_tensor.get_outer_size()[dim_idx];
  }
  this->macro_loop_idx = new TensorIdx [tensor_dim];
  this->remainder_loop_idx = new TensorIdx [tensor_dim];
}


template <typename FloatType>
ParamTrans<FloatType>::ParamTrans(const ParamTrans &param_obj) {
    : input_tensor(param_obj.input_tensor),
      output_tensor(param_obj.output_tensor),
      alpha(param_obj.alpha),
      beta(param_obj.beta),
      tensor_dim(param_obj.tensor_dim),
      input_offset(param_obj.input_tensor),
      output_offset(param_obj.output_tensor),
      perm(param_obj.perm) {
  this->macro_loop_idx = new TensorIdx [tensor_dim];
  std::copy(param_obj.macro_loop_idx, param_obj.macro_loop_idx + tensor_dim,
      this->macro_loop_idx);
  this->remainder_loop_idx = new TensorIdx [tensor_dim];
  std::copy(param_obj.remainder_loop_idx,
      param_obj.remainder_loop_idx + tensor_dim, this->remainder_loop_idx);
}


template <typename FloatType>
ParamTrans<FloatType>::ParamTrans(ParamTrans &&param_obj) noexcept
    : input_tensor(param_obj.input_tensor),
      output_tensor(param_obj.output_tensor),
      alpha(param_obj.alpha),
      beta(param_obj.beta),
      tensor_dim(param_obj.tensor_dim),
      input_offset(param_obj.input_tensor),
      output_offset(param_obj.output_tensor),
      perm(std::move(param_obj.perm)),
      macro_loop_idx(param_obj.macro_loop_idx),
      remainder_loop_idx(param_obj.remainder_loop_idx) {
  param_obj.macro_loop_idx = nullptr;
  param_obj.remainder_loop_idx = nullptr;
}


template <typename FloatType>
ParamTrans<FloatType>::~ParamTrans() {
  delete [] this->macro_loop_idx;
  delete [] this->remainder_loop_idx;
}

}

#endif // HPTC_PARAMETER_TRANS_H_
