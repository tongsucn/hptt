#pragma once
#ifndef HPTT_PARAM_PARAMETER_TRANS_TCC_
#define HPTT_PARAM_PARAMETER_TRANS_TCC_

/*
 * Implementation for class TensorMergedWrapper
 */
template <typename FloatType,
          TensorUInt ORDER>
TensorMergedWrapper<FloatType, ORDER>::TensorMergedWrapper(
    const TensorWrapper<FloatType, ORDER> &tensor,
    const std::unordered_set<TensorUInt> &merge_set)
    : TensorWrapper<FloatType, ORDER>(tensor),
      begin_order_idx_(0),
      merged_order_(this->merge_idx_(merge_set)) {
}


template <typename FloatType,
          TensorUInt ORDER>
HPTT_INL FloatType &TensorMergedWrapper<FloatType, ORDER>::operator[](
    const TensorIdx * RESTRICT indices) {
  TensorIdx abs_offset = 0;
  for (TensorUInt idx = this->begin_order_idx_; idx < ORDER; ++idx)
    abs_offset += indices[idx] * this->strides_[idx];
  return this->raw_data_[abs_offset];
}


template <typename FloatType,
          TensorUInt ORDER>
HPTT_INL const FloatType &TensorMergedWrapper<FloatType, ORDER>::operator[](
    const TensorIdx * RESTRICT indices) const {
  TensorIdx abs_offset = 0;
  for (TensorUInt idx = this->begin_order_idx_; idx < ORDER; ++idx)
    abs_offset += indices[idx] * this->strides_[idx];
  return this->raw_data_[abs_offset];
}


template <typename FloatType,
          TensorUInt ORDER>
HPTT_INL FloatType &TensorMergedWrapper<FloatType, ORDER>::operator[](
    TensorIdx **indices) {
  TensorIdx abs_offset = 0;
  for (TensorUInt idx = this->begin_order_idx_; idx < ORDER; ++idx)
    abs_offset += *indices[idx] * this->strides_[idx];
  return this->raw_data_[abs_offset];
}


template <typename FloatType,
          TensorUInt ORDER>
HPTT_INL const FloatType &TensorMergedWrapper<FloatType, ORDER>::operator[](
    const TensorIdx **indices) const {
  TensorIdx abs_offset = 0;
  for (TensorUInt idx = this->begin_order_idx_; idx < ORDER; ++idx)
    abs_offset += *indices[idx] * this->strides_[idx];
  return this->raw_data_[abs_offset];
}


template <typename FloatType,
          TensorUInt ORDER>
TensorUInt TensorMergedWrapper<FloatType, ORDER>::merge_idx_(
    const std::unordered_set<TensorUInt> &merge_set) {
  if (ORDER <= 2)
    return ORDER;

  // Merge size and outer size
  for (TensorInt idx = ORDER - 1, curr_idx = ORDER; idx >= 0; --idx) {
    if (1 == merge_set.count(idx)) {
      --curr_idx;
      this->size_[curr_idx] = this->size_[idx];
      this->outer_size_[curr_idx] = this->outer_size_[idx];
    }
    else {
      this->size_[curr_idx] *= this->size_[idx];
      this->outer_size_[curr_idx] *= this->outer_size_[idx];
    }
  }

  // Merge strides
  this->begin_order_idx_ = ORDER - merge_set.size();
  this->strides_[this->begin_order_idx_] = 1;
  for (TensorUInt idx = this->begin_order_idx_; idx < ORDER - 1; ++idx)
    this->strides_[idx + 1] = this->outer_size_[idx] * this->strides_[idx];

  // Fill in unused part
  for (TensorUInt idx = 0; idx < this->begin_order_idx_; ++idx) {
    this->size_[idx] = 1;
    this->outer_size_[idx] = 1;
  }
  std::fill(this->strides_, this->strides_ + this->begin_order_idx_, 0);

  return merge_set.size();
}


/*
 * Implementation for struct ParamTrans
 */
template <typename TensorType,
          bool UPDATE_OUT>
ParamTrans<TensorType, UPDATE_OUT>::ParamTrans(const TensorType &input_tensor,
    TensorType &output_tensor, const std::array<TensorUInt, ORDER> &perm,
    const DeducedFloatType<typename TensorType::Float> alpha,
    const DeducedFloatType<typename TensorType::Float> beta)
    : input_merge_set_(), output_merge_set_(), kn_(), perm(perm),
      stride_in_inld(1), stride_in_outld(1),
      stride_out_inld(1), stride_out_outld(1),
      merged_order(this->merge_idx_(input_tensor, output_tensor, perm)),
      input_tensor(input_tensor, this->input_merge_set_),
      output_tensor(output_tensor, this->output_merge_set_) {
  // Initialize coefficients
  this->set_coef(alpha, beta);

  // Initialize access strides
  if (this->is_common_leading()) {
    // Common leading case
    // stride of input tensor's 2nd order in input tensor
    this->stride_in_inld = this->input_tensor.get_outer_size(
        this->begin_order_idx);

    // stride of output tensor's 2nd-order in input tensor
    const auto in_outld_end
        = this->perm[this->begin_order_idx + 1] + this->begin_order_idx;
    for (auto order_idx = this->begin_order_idx; order_idx != in_outld_end;
        ++order_idx)
      this->stride_in_outld *= this->input_tensor.get_outer_size(order_idx);

    // stride of input tensor's 2nd-order in output tensor
    for (auto order_idx = this->begin_order_idx; 1 != this->perm[order_idx];
        ++order_idx)
      this->stride_out_inld *= this->output_tensor.get_outer_size(order_idx);

    // stride of output tensor's 2nd-order in output tensor
    this->stride_out_outld = this->output_tensor.get_outer_size(
        this->begin_order_idx);
  }
  else {
    // Non-common leading case
    for (auto order_idx = this->begin_order_idx;
        order_idx != this->perm[this->begin_order_idx] + this->begin_order_idx;
        ++order_idx)
      this->stride_in_outld *= this->input_tensor.get_outer_size(order_idx);
    for (auto order_idx = this->begin_order_idx;
        0 != this->perm[order_idx]; ++order_idx)
      this->stride_out_inld *= this->output_tensor.get_outer_size(order_idx);
  }
}


template <typename TensorType,
          bool UPDATE_OUT>
HPTT_INL bool ParamTrans<TensorType, UPDATE_OUT>::is_common_leading() const {
  return 0 == this->perm[this->begin_order_idx];
}


template <typename TensorType,
          bool UPDATE_OUT>
HPTT_INL void ParamTrans<TensorType, UPDATE_OUT>::set_coef(
    const DeducedFloatType<typename TensorType::Float> alpha,
    const DeducedFloatType<typename TensorType::Float> beta) {
  this->alpha = alpha, this->beta = beta;
  this->kn_.set_coef(alpha, beta);
}


template <typename TensorType,
          bool UPDATE_OUT>
HPTT_INL const KernelPackTrans<typename TensorType::Float, UPDATE_OUT> &
ParamTrans<TensorType, UPDATE_OUT>::get_kernel() const {
  return this->kn_;
}


template <typename TensorType,
          bool UPDATE_OUT>
void ParamTrans<TensorType, UPDATE_OUT>::set_lin_wrapper_loop(
    const TensorUInt size_kn_inld, const TensorUInt size_kn_outld) {
  this->kn_.kn_lin_core.set_wrapper_loop(this->stride_in_inld,
      this->stride_in_outld, this->stride_out_inld, this->stride_out_outld,
      size_kn_inld, size_kn_outld);
  this->kn_.kn_lin_right.set_wrapper_loop(this->stride_in_inld,
      this->stride_in_outld, this->stride_out_inld, this->stride_out_outld, 1,
      size_kn_outld);
  this->kn_.kn_lin_bottom.set_wrapper_loop(this->stride_in_inld,
      this->stride_in_outld, this->stride_out_inld, this->stride_out_outld,
      size_kn_inld, 1);
}


template <typename TensorType,
          bool UPDATE_OUT>
void ParamTrans<TensorType, UPDATE_OUT>::set_sca_wrapper_loop(
    const TensorUInt size_kn_in_inld, const TensorUInt size_kn_in_outld,
    const TensorUInt size_kn_out_inld, const TensorUInt size_kn_out_outld) {
  this->kn_.kn_sca_right.set_wrapper_loop(this->stride_in_inld,
      this->stride_in_outld, this->stride_out_inld, this->stride_out_outld,
      size_kn_in_inld, size_kn_in_outld);
  this->kn_.kn_sca_bottom.set_wrapper_loop(this->stride_in_inld,
      this->stride_in_outld, this->stride_out_inld, this->stride_out_outld,
      size_kn_out_inld, size_kn_out_outld);
  this->kn_.kn_sca_scalar.set_wrapper_loop(this->stride_in_inld,
      this->stride_in_outld, this->stride_out_inld, this->stride_out_outld,
      size_kn_in_inld, size_kn_out_outld);
}


template <typename TensorType,
          bool UPDATE_OUT>
void ParamTrans<TensorType, UPDATE_OUT>::reset_data(const Float *data_in,
    Float *data_out) {
  this->input_tensor.reset_data(data_in);
  this->output_tensor.reset_data(data_out);
}


template <typename TensorType,
          bool UPDATE_OUT>
TensorUInt ParamTrans<TensorType, UPDATE_OUT>::merge_idx_(
    const TensorType &input_tensor, const TensorType &output_tensor,
    const std::array<TensorUInt, ORDER> &perm) {
  if (ORDER <= 1)
    return ORDER;

  const auto &input_size = input_tensor.get_size();
  const auto &input_outer_size = input_tensor.get_outer_size();
  const auto &output_size = output_tensor.get_size();
  const auto &output_outer_size = output_tensor.get_outer_size();

  // Create permutation set
  for (TensorUInt order_idx = 1; order_idx < ORDER; ++order_idx) {
    // If current order ID does not equal to previous order ID plus one, or
    // the previous order size does not equal to the outer size, then push
    // previous ID into set.
    if (perm[order_idx] != perm[order_idx - 1] + 1 or
        input_size[perm[order_idx - 1]] != input_outer_size[perm[order_idx - 1]]
        or output_size[order_idx - 1] != output_outer_size[order_idx - 1]) {
      this->input_merge_set_.insert(perm[order_idx - 1]);
      this->output_merge_set_.insert(order_idx - 1);
    }
  }

  // input/output merge sets are initialized, they are ready for initializing
  // TensorMergedWrapper
  this->input_merge_set_.insert(perm[ORDER - 1]);
  this->output_merge_set_.insert(ORDER - 1);

  // Check merged order
  const auto merged = static_cast<TensorUInt>(this->input_merge_set_.size());
  this->begin_order_idx = ORDER - merged;
  if (ORDER == merged)
    return ORDER;

  // Update permutation array
  // Create an array for storing sorted keys in input_merge_set_,
  TensorUInt sorted_perm_arr[ORDER];
  std::copy(this->input_merge_set_.begin(), this->input_merge_set_.end(),
      sorted_perm_arr);
  std::sort(sorted_perm_arr, sorted_perm_arr + merged);

  // Create an unordered map to store the mapping from original order ID to
  // updated order ID.
  std::unordered_map<TensorUInt, TensorUInt> perm_map;
  for (TensorUInt order_idx = 0; order_idx < merged; ++order_idx)
    perm_map[sorted_perm_arr[order_idx]] = order_idx;

  // Update permutation array
  for (TensorInt order_idx = ORDER - 1, update_idx = ORDER - 1;
      order_idx >= 0; --order_idx) {
    if (1 == this->input_merge_set_.count(this->perm[order_idx])) {
      this->perm[update_idx] = perm_map[this->perm[order_idx]];
      --update_idx;
    }
  }
  // Fill unused part of the permutation array
  std::fill(this->perm.begin(), this->perm.begin() + this->begin_order_idx, 0);

  return merged;
}


/*
 * Import explicit instantiation declaration for stract ParamTrans, this file
 * should be generated by cmake script.
 */
#include <hptt/gen/parameter_trans_gen.tcc>

#endif // HPTT_PARAM_PARAMETER_TRANS_TCC_
