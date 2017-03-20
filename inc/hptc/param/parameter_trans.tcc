#pragma once
#ifndef HPTC_PARAM_PARAMETER_TRANS_TCC_
#define HPTC_PARAM_PARAMETER_TRANS_TCC_

/*
 * Implementation for class TensorMergedWrapper
 */
template <typename FloatType,
          TensorOrder ORDER>
template <MemLayout ACT_MAJOR>
TensorMergedWrapper<FloatType, ORDER>::TensorMergedWrapper(
    TensorWrapper<FloatType, ORDER, ACT_MAJOR> &tensor)
    : TensorWrapper<FloatType, ORDER, MemLayout::COL_MAJOR>(tensor),
      merged_order_(ORDER) {
}


template <typename FloatType,
          TensorOrder ORDER>
INLINE FloatType &TensorMergedWrapper<FloatType, ORDER>::operator[](
    const TensorIdx * RESTRICT indices) {
  TensorIdx abs_offset = 0;
  for (TensorIdx idx = ORDER - this->merged_order_; idx < ORDER; ++idx)
    abs_offset += (this->offsets_[idx] + indices[idx]) * this->strides_[idx];
  return this->raw_data_[abs_offset];
}


template <typename FloatType,
          TensorOrder ORDER>
INLINE const FloatType &TensorMergedWrapper<FloatType, ORDER>::operator[](
    const TensorIdx * RESTRICT indices) const {
  TensorIdx abs_offset = 0;
  for (TensorIdx idx = ORDER - this->merged_order_; idx < ORDER; ++idx)
    abs_offset += (this->offsets_[idx] + indices[idx]) * this->strides_[idx];
  return this->raw_data_[abs_offset];
}


template <typename FloatType,
          TensorOrder ORDER>
INLINE FloatType &TensorMergedWrapper<FloatType, ORDER>::operator[](
    TensorIdx **indices) {
  TensorIdx abs_offset = 0;
  for (TensorIdx idx = ORDER - this->merged_order_; idx < ORDER; ++idx)
    abs_offset += (this->offsets_[idx] + *indices[idx]) * this->strides_[idx];
  return this->raw_data_[abs_offset];
}


template <typename FloatType,
          TensorOrder ORDER>
INLINE const FloatType &TensorMergedWrapper<FloatType, ORDER>::operator[](
    const TensorIdx **indices) const {
  TensorIdx abs_offset = 0;
  for (TensorIdx idx = ORDER - this->merged_order_; idx < ORDER; ++idx)
    abs_offset += (this->offsets_[idx] + *indices[idx]) * this->strides_[idx];
  return this->raw_data_[abs_offset];
}


template <typename FloatType,
          TensorOrder ORDER>
void TensorMergedWrapper<FloatType, ORDER>::merge_idx(
    const std::unordered_set<TensorOrder> &merge_set) {
  if (ORDER <= 2)
    return;

  this->merged_order_ = static_cast<TensorOrder>(merge_set.size());
  const TensorIdx start_idx = ORDER - this->merged_order_;

  // Merge size, outer size and offsets
  for (TensorIdx idx = ORDER - 1, curr_idx = ORDER; idx >= 0; --idx) {
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
  this->strides_[start_idx] = 1;
  for (TensorIdx idx = start_idx; idx < ORDER - 1; ++idx)
    this->strides_[idx + 1] = this->outer_size_[idx] * this->strides_[idx];

  // Fill the unused part
  for (TensorIdx idx = 0; idx < start_idx; ++idx) {
    this->size_[idx] = 1;
    this->outer_size_[idx] = 1;
  }
  std::fill(this->offsets_, this->offsets_ + start_idx, 0);
  std::fill(this->strides_, this->strides_ + start_idx, 0);
}


/*
 * Implementation for struct ParamTrans
 */
template <typename TensorType,
          CoefUsageTrans USAGE>
ParamTrans<TensorType, USAGE>::ParamTrans(TensorType &input_tensor,
    TensorType &output_tensor, const std::array<TensorOrder, ORDER> &perm,
    const DeducedFloatType<typename TensorType::FLOAT> alpha,
    const DeducedFloatType<typename TensorType::FLOAT> beta)
    : input_tensor(input_tensor), output_tensor(output_tensor), perm(perm),
      alpha(alpha), beta(beta),
      input_stride(1), output_stride(1),
      merged_order(this->merge_idx_(perm)),
      begin_order_idx(ORDER - this->merged_order),
      kn(KernelPackTrans<FloatType, USAGE>::get_package()) {
  // Initialize registers
  using FloatType = typename TensorType::FLOAT;
  using KernelPack = KernelPackTrans<FloatType, USAGE>;

  this->reg_alpha_full = KernelPack::reg_coef_full(this->alpha);
  this->reg_beta_full = KernelPack::reg_coef_full(this->beta);
  this->reg_alpha_half = KernelPack::reg_coef_half(this->alpha);
  this->reg_beta_half = KernelPack::reg_coef_half(this->beta);
  this->reg_alpha_linear = KernelPack::reg_coef_linear(this->alpha);
  this->reg_beta_linear = KernelPack::reg_coef_linear(this->beta);

  // Initialize access strides
  for (TensorIdx idx = 0; idx < perm[0]; ++idx)
    this->input_stride *= input_tensor.get_outer_size()[idx];
  for (TensorIdx idx = 0; 0 != perm[idx]; ++idx)
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
INLINE std::pair<TensorOrder, TensorOrder>
ParamTrans<TensorType, USAGE>::get_leading() {
  std::pair<TensorOrder, TensorOrder> result;

  result.first = this->input_tensor.get_size()[this->begin_order_idx];
  result.second = this->output_tensor.get_size()[this->begin_order_idx];

  return result;
}


template <typename TensorType,
          CoefUsageTrans USAGE>
INLINE void ParamTrans<TensorType, USAGE>::set_coef(
    const DeducedFloatType<typename TensorType::FLOAT> alpha,
    const DeducedFloatType<typename TensorType::FLOAT> beta) {
  using FloatType = typename TensorType::FLOAT;
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
TensorOrder ParamTrans<TensorType, USAGE>::merge_idx_(
    const std::array<TensorOrder, ORDER> &perm) {
  if (ORDER <= 1)
    return ORDER;

  const auto &input_size = this->input_tensor.get_size();
  const auto &input_outer_size = this->input_tensor.get_outer_size();
  const auto &output_size = this->output_tensor.get_size();
  const auto &output_outer_size = this->output_tensor.get_outer_size();
  std::unordered_set<TensorOrder> input_perm_set, output_perm_set;

  // Create permutation set
  for (TensorOrder idx = 1; idx < ORDER; ++idx) {
    // If current order ID does not equal to previous order ID plus one, or
    // the previous order size does not equal to the outer size, then push
    // previous ID into set.
    if (perm[idx] != perm[idx - 1] + 1 or
        input_size[perm[idx - 1]] != input_outer_size[perm[idx - 1]] or
        output_size[idx - 1] != output_outer_size[idx - 1]) {
      input_perm_set.insert(perm[idx - 1]);
      output_perm_set.insert(idx - 1);
    }
  }
  input_perm_set.insert(perm[ORDER - 1]);
  output_perm_set.insert(ORDER - 1);

  // Set merged order
  auto merged = static_cast<TensorOrder>(input_perm_set.size());
  if (ORDER == merged)
    return ORDER;

  // Update permutation array
  // Create an array for storing sorted keys in input_perm_set,
  TensorOrder sorted_perm_arr[ORDER];
  std::copy(input_perm_set.begin(), input_perm_set.end(), sorted_perm_arr);
  std::sort(sorted_perm_arr, sorted_perm_arr + merged);

  // Create an unordered map to store the mapping from original order ID to
  // updated order ID.
  std::unordered_map<TensorOrder, TensorOrder> perm_map;
  for (TensorIdx idx = 0; idx < merged; ++idx)
    perm_map[sorted_perm_arr[idx]] = idx;

  // Update permutation array
  for (TensorIdx idx = ORDER - 1, curr_idx = ORDER - 1; idx >= 0; --idx) {
    if (1 == input_perm_set.count(this->perm[idx])) {
      this->perm[curr_idx] = perm_map[this->perm[idx]];
      --curr_idx;
    }
  }
  // Fill unused part of the permutation array
  std::fill(this->perm.begin(), this->perm.begin() + this->begin_order_idx, 0);

  // Execute merge
  this->input_tensor.merge_idx(input_perm_set);
  this->output_tensor.merge_idx(output_perm_set);

  return merged;
}


/*
 * Import explicit instantiation declaration for stract ParamTrans, this file
 * should be generated by cmake script.
 */
#include "parameter_trans_gen.tcc"

#endif // HPTC_PARAM_PARAMETER_TRANS_TCC_
