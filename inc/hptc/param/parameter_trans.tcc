#pragma once
#ifndef HPTC_PARAM_PARAMETER_TRANS_TCC_
#define HPTC_PARAM_PARAMETER_TRANS_TCC_

/*
 * Implementation for class TensorMergedWrapper
 */
template <typename FloatType,
          TensorUInt ORDER>
template <MemLayout ACT_MAJOR>
TensorMergedWrapper<FloatType, ORDER>::TensorMergedWrapper(
    const TensorWrapper<FloatType, ORDER, ACT_MAJOR> &tensor,
    const std::unordered_set<TensorUInt> &merge_set)
    : TensorWrapper<FloatType, ORDER, MemLayout::COL_MAJOR>(tensor),
      begin_order_idx_(0),
      merged_order_(this->merge_idx_(merge_set)) {
}


template <typename FloatType,
          TensorUInt ORDER>
HPTC_INL FloatType &TensorMergedWrapper<FloatType, ORDER>::operator[](
    const TensorIdx * RESTRICT indices) {
  TensorIdx abs_offset = 0;
  for (TensorUInt idx = this->begin_order_idx_; idx < ORDER; ++idx)
    abs_offset += indices[idx] * this->strides_[idx];
  return this->raw_data_[abs_offset];
}


template <typename FloatType,
          TensorUInt ORDER>
HPTC_INL const FloatType &TensorMergedWrapper<FloatType, ORDER>::operator[](
    const TensorIdx * RESTRICT indices) const {
  TensorIdx abs_offset = 0;
  for (TensorUInt idx = this->begin_order_idx_; idx < ORDER; ++idx)
    abs_offset += indices[idx] * this->strides_[idx];
  return this->raw_data_[abs_offset];
}


template <typename FloatType,
          TensorUInt ORDER>
HPTC_INL FloatType &TensorMergedWrapper<FloatType, ORDER>::operator[](
    TensorIdx **indices) {
  TensorIdx abs_offset = 0;
  for (TensorUInt idx = this->begin_order_idx_; idx < ORDER; ++idx)
    abs_offset += *indices[idx] * this->strides_[idx];
  return this->raw_data_[abs_offset];
}


template <typename FloatType,
          TensorUInt ORDER>
HPTC_INL const FloatType &TensorMergedWrapper<FloatType, ORDER>::operator[](
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
template <typename TensorType>
ParamTrans<TensorType>::ParamTrans(const TensorType &input_tensor,
    TensorType &output_tensor, const std::array<TensorUInt, ORDER> &perm,
    const DeducedFloatType<typename TensorType::Float> alpha,
    const DeducedFloatType<typename TensorType::Float> beta)
    : input_merge_set_(), output_merge_set_(), kn_(), perm(perm),
      input_stride(1), output_stride(1),
      merged_order(this->merge_idx_(perm)),
      input_tensor(input_tensor, this->input_merge_set_),
      output_tensor(output_tensor, this->output_merge_set_) {
  this->set_coef(alpha, beta);
  // Initialize access strides
  for (TensorUInt order_idx = 0; order_idx < perm[0]; ++order_idx)
    this->input_stride *= input_tensor.get_outer_size(order_idx);
  for (TensorUInt order_idx = 0; 0 != perm[order_idx]; ++order_idx)
    this->output_stride *= output_tensor.get_outer_size(order_idx);
}


template <typename TensorType>
HPTC_INL bool ParamTrans<TensorType>::is_common_leading() const {
  return 0 == this->perm[this->begin_order_idx];
}


template <typename TensorType>
HPTC_INL std::pair<TensorUInt, TensorUInt>
ParamTrans<TensorType>::get_leading() const {
  return std::make_pair<TensorUInt, TensorUInt>(
      this->input_tensor.get_size(this->begin_order_idx),
      this->input_tensor.get_size(this->perm[this->begin_order_idx]));
}


template <typename TensorType>
HPTC_INL void ParamTrans<TensorType>::set_coef(
    const DeducedFloatType<typename TensorType::Float> alpha,
    const DeducedFloatType<typename TensorType::Float> beta) {
  this->alpha = alpha, this->beta = beta;
  this->kn_.set_coef(alpha, beta);
}


template <typename TensorType>
HPTC_INL const KernelPackTrans<typename TensorType::Float> &
ParamTrans<TensorType>::get_kernel() const {
  return this->kn_;
}


template <typename TensorType>
TensorUInt ParamTrans<TensorType>::merge_idx_(
    const std::array<TensorUInt, ORDER> &perm) {
  if (ORDER <= 1)
    return ORDER;

  const auto &input_size = this->input_tensor.get_size();
  const auto &input_outer_size = this->input_tensor.get_outer_size();
  const auto &output_size = this->output_tensor.get_size();
  const auto &output_outer_size = this->output_tensor.get_outer_size();

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
  auto merged = static_cast<TensorUInt>(this->input_merge_set_.size());
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
#include <hptc/gen/parameter_trans_gen.tcc>

#endif // HPTC_PARAM_PARAMETER_TRANS_TCC_
