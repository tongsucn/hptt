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
      merged_order_(this->merge_idx_(merge_set)) {
}


template <typename FloatType,
          TensorUInt ORDER>
INLINE FloatType &TensorMergedWrapper<FloatType, ORDER>::operator[](
    const TensorIdx * RESTRICT indices) {
  TensorIdx abs_offset = 0;
  for (auto idx = ORDER - this->merged_order_; idx < ORDER; ++idx)
    abs_offset += (this->offsets_[idx] + indices[idx]) * this->strides_[idx];
  return this->raw_data_[abs_offset];
}


template <typename FloatType,
          TensorUInt ORDER>
INLINE const FloatType &TensorMergedWrapper<FloatType, ORDER>::operator[](
    const TensorIdx * RESTRICT indices) const {
  TensorIdx abs_offset = 0;
  for (auto idx = ORDER - this->merged_order_; idx < ORDER; ++idx)
    abs_offset += (this->offsets_[idx] + indices[idx]) * this->strides_[idx];
  return this->raw_data_[abs_offset];
}


template <typename FloatType,
          TensorUInt ORDER>
INLINE FloatType &TensorMergedWrapper<FloatType, ORDER>::operator[](
    TensorIdx **indices) {
  TensorIdx abs_offset = 0;
  for (auto idx = ORDER - this->merged_order_; idx < ORDER; ++idx)
    abs_offset += (this->offsets_[idx] + *indices[idx]) * this->strides_[idx];
  return this->raw_data_[abs_offset];
}


template <typename FloatType,
          TensorUInt ORDER>
INLINE const FloatType &TensorMergedWrapper<FloatType, ORDER>::operator[](
    const TensorIdx **indices) const {
  TensorIdx abs_offset = 0;
  for (auto idx = ORDER - this->merged_order_; idx < ORDER; ++idx)
    abs_offset += (this->offsets_[idx] + *indices[idx]) * this->strides_[idx];
  return this->raw_data_[abs_offset];
}


template <typename FloatType,
          TensorUInt ORDER>
TensorUInt TensorMergedWrapper<FloatType, ORDER>::merge_idx_(
    const std::unordered_set<TensorUInt> &merge_set) {
  if (ORDER <= 2)
    return ORDER;

  // Merge size, outer size and offsets
  for (TensorInt idx = ORDER - 1, curr_idx = ORDER; idx >= 0; --idx) {
    if (1 == merge_set.count(idx)) {
      --curr_idx;
      this->size_[curr_idx] = this->size_[idx];
      this->outer_size_[curr_idx] = this->outer_size_[idx];
      this->offsets_[curr_idx] = this->offsets_[idx];
    }
    else {
      this->size_[curr_idx] *= this->size_[idx];
      this->outer_size_[curr_idx] *= this->outer_size_[idx];
    }
  }

  // Merge strides
  const auto begin_order_idx = ORDER - merge_set.size();
  this->strides_[begin_order_idx] = 1;
  for (auto idx = begin_order_idx; idx < ORDER - 1; ++idx)
    this->strides_[idx + 1] = this->outer_size_[idx] * this->strides_[idx];

  // Fill the unused part
  for (auto idx = 0; idx < begin_order_idx; ++idx) {
    this->size_[idx] = 1;
    this->outer_size_[idx] = 1;
  }
  std::fill(this->offsets_, this->offsets_ + begin_order_idx, 0);
  std::fill(this->strides_, this->strides_ + begin_order_idx, 0);

  return merge_set.size();
}


/*
 * Implementation for struct ParamTrans
 */
template <typename TensorType,
          CoefUsageTrans USAGE>
ParamTrans<TensorType, USAGE>::ParamTrans(const TensorType &input_tensor,
    TensorType &output_tensor, const std::array<TensorUInt, ORDER> &perm,
    const DeducedFloatType<typename TensorType::FloatType> alpha,
    const DeducedFloatType<typename TensorType::FloatType> beta)
    : input_merge_set_(), output_merge_set_(),
      perm(perm), alpha(alpha), beta(beta),
      input_stride(1), output_stride(1),
      merged_order(this->merge_idx_(perm)),
      begin_order_idx(ORDER - this->merged_order),
      input_tensor(input_tensor, this->input_merge_set_),
      output_tensor(output_tensor, this->output_merge_set_),
      kn(KernelPackTrans<FloatType, USAGE>::get_package()) {
  // Initialize registers
  using FloatType = typename TensorType::FloatType;
  using KernelPack = KernelPackTrans<FloatType, USAGE>;

  this->reg_alpha_full = KernelPack::reg_coef_full(this->alpha);
  this->reg_beta_full = KernelPack::reg_coef_full(this->beta);
  this->reg_alpha_half = KernelPack::reg_coef_half(this->alpha);
  this->reg_beta_half = KernelPack::reg_coef_half(this->beta);
  this->reg_alpha_linear = KernelPack::reg_coef_linear(this->alpha);
  this->reg_beta_linear = KernelPack::reg_coef_linear(this->beta);

  // Initialize access strides
  for (auto idx = 0; idx < perm[0]; ++idx)
    this->input_stride *= input_tensor.get_outer_size()[idx];
  for (auto idx = 0; 0 != perm[idx]; ++idx)
    this->output_stride *= output_tensor.get_outer_size()[idx];
}


template <typename TensorType,
          CoefUsageTrans USAGE>
INLINE bool ParamTrans<TensorType, USAGE>::is_common_leading() {
  if (0 == this->perm[this->begin_order_idx])
    return true;
  return false;
}


template <typename TensorType,
          CoefUsageTrans USAGE>
INLINE std::pair<TensorUInt, TensorUInt>
ParamTrans<TensorType, USAGE>::get_leading() {
  std::pair<TensorUInt, TensorUInt> result;

  result.first = this->input_tensor.get_size()[this->begin_order_idx];
  result.second = this->output_tensor.get_size()[this->begin_order_idx];

  return result;
}


template <typename TensorType,
          CoefUsageTrans USAGE>
INLINE void ParamTrans<TensorType, USAGE>::set_coef(
    const DeducedFloatType<typename TensorType::FloatType> alpha,
    const DeducedFloatType<typename TensorType::FloatType> beta) {
  using FloatType = typename TensorType::FloatType;
  using KernelPack = KernelPackTrans<FloatType, USAGE>;

  this->alpha = alpha, this->beta = beta;
  this->reg_alpha_full = KernelPack::reg_coef_full(this->alpha);
  this->reg_beta_full = KernelPack::reg_coef_full(this->beta);
  this->reg_alpha_half = KernelPack::reg_coef_half(this->alpha);
  this->reg_beta_half = KernelPack::reg_coef_half(this->beta);
  this->reg_alpha_linear = KernelPack::reg_coef_linear(this->alpha);
  this->reg_beta_linear = KernelPack::reg_coef_linear(this->beta);
}


template <typename TensorType,
          CoefUsageTrans USAGE>
TensorUInt ParamTrans<TensorType, USAGE>::merge_idx_(
    const std::array<TensorUInt, ORDER> &perm) {
  if (ORDER <= 1)
    return ORDER;

  const auto &input_size = this->input_tensor.get_size();
  const auto &input_outer_size = this->input_tensor.get_outer_size();
  const auto &output_size = this->output_tensor.get_size();
  const auto &output_outer_size = this->output_tensor.get_outer_size();

  // Create permutation set
  for (auto idx = 1; idx < ORDER; ++idx) {
    // If current order ID does not equal to previous order ID plus one, or
    // the previous order size does not equal to the outer size, then push
    // previous ID into set.
    if (perm[idx] != perm[idx - 1] + 1 or
        input_size[perm[idx - 1]] != input_outer_size[perm[idx - 1]] or
        output_size[idx - 1] != output_outer_size[idx - 1]) {
      this->input_merge_set_.insert(perm[idx - 1]);
      this->output_merge_set_.insert(idx - 1);
    }
  }
  this->input_merge_set_.insert(perm[ORDER - 1]);
  this->output_merge_set_.insert(ORDER - 1);

  // Set merged order
  auto merged = static_cast<TensorUInt>(this->input_merge_set_.size());
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
  for (auto idx = 0; idx < merged; ++idx)
    perm_map[sorted_perm_arr[idx]] = idx;

  // Update permutation array
  for (TensorIdx idx = ORDER - 1, curr_idx = ORDER - 1; idx >= 0; --idx) {
    if (1 == this->input_merge_set_.count(this->perm[idx])) {
      this->perm[curr_idx] = perm_map[this->perm[idx]];
      --curr_idx;
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
