#pragma once
#ifndef HPTC_PARAM_PARAMETER_TRANS_TCC_
#define HPTC_PARAM_PARAMETER_TRANS_TCC_

/*
 * Implementation for class TensorMergedWrapper
 */
template <typename FloatType,
          TensorOrder ORDER,
          MemLayout LAYOUT>
TensorMergedWrapper<FloatType, ORDER, LAYOUT>::TensorMergedWrapper(
    const TensorWrapper<FloatType, ORDER, LAYOUT> &wrapper)
    : TensorWrapper<FloatType, ORDER, LAYOUT>(wrapper),
      merged_order_(ORDER) {
}


template <typename FloatType,
          TensorOrder ORDER,
          MemLayout LAYOUT>
INLINE FloatType &TensorMergedWrapper<FloatType, ORDER, LAYOUT>::operator[](
    const TensorIdx * RESTRICT indices) {
  TensorIdx abs_offset = 0;
  for (TensorIdx idx = ORDER - this->merged_order_; idx < ORDER; ++idx)
    abs_offset += (this->offsets_[idx] + indices[idx]) * this->strides_[idx];
  return this->raw_data_[abs_offset];
}


template <typename FloatType,
          TensorOrder ORDER,
          MemLayout LAYOUT>
INLINE const FloatType &
TensorMergedWrapper<FloatType, ORDER, LAYOUT>::operator[](
    const TensorIdx * RESTRICT indices) const {
  TensorIdx abs_offset = 0;
  for (TensorIdx idx = ORDER - this->merged_order_; idx < ORDER; ++idx)
    abs_offset += (this->offsets_[idx] + indices[idx]) * this->strides_[idx];
  return this->raw_data_[abs_offset];
}


template <typename FloatType,
          TensorOrder ORDER,
          MemLayout LAYOUT>
INLINE FloatType &TensorMergedWrapper<FloatType, ORDER, LAYOUT>::operator[](
    TensorIdx **indices) {
  TensorIdx abs_offset = 0;
  for (TensorIdx idx = ORDER - this->merged_order_; idx < ORDER; ++idx)
    abs_offset += (this->offsets_[idx] + *indices[idx]) * this->strides_[idx];
  return this->raw_data_[abs_offset];
}


template <typename FloatType,
          TensorOrder ORDER,
          MemLayout LAYOUT>
INLINE const FloatType &
TensorMergedWrapper<FloatType, ORDER, LAYOUT>::operator[](
    const TensorIdx **indices) const {
  TensorIdx abs_offset = 0;
  for (TensorIdx idx = ORDER - this->merged_order_; idx < ORDER; ++idx)
    abs_offset += (this->offsets_[idx] + *indices[idx]) * this->strides_[idx];
  return this->raw_data_[abs_offset];
}


template <typename FloatType,
          TensorOrder ORDER,
          MemLayout LAYOUT>
INLINE TensorOrder
TensorMergedWrapper<FloatType, ORDER, LAYOUT>::get_leading() {
  auto idx = ORDER - (MemLayout::COL_MAJOR == LAYOUT ? this->merged_order_ : 1);
  return this->size_[idx];
}


template <typename FloatType,
          TensorOrder ORDER,
          MemLayout LAYOUT>
void TensorMergedWrapper<FloatType, ORDER, LAYOUT>::merge_idx(
    const std::unordered_set<TensorOrder> &merge_set) {
  if (ORDER <= 2)
    return;

  this->merged_order_ = static_cast<TensorOrder>(merge_set.size());
  const TensorIdx start_idx = ORDER - this->merged_order_;

  if (MemLayout::COL_MAJOR == LAYOUT) {
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
  }
  else {
    // Merge size, outer size and offsets
    TensorIdx size_acc = 1, outer_size_acc = 1;
    for (TensorIdx idx = ORDER - 1, curr_idx = ORDER - 1; idx >= 0; --idx) {
      if (1 == merge_set.count(idx)) {
        this->size_[curr_idx] = this->size_[idx] * size_acc;
        this->outer_size_[curr_idx] = this->outer_size_[idx] * outer_size_acc;
        this->offsets_[curr_idx] = this->offsets_[idx];
        size_acc = outer_size_acc = 1;
        --curr_idx;
      }
      else {
        size_acc *= this->size_[idx];
        outer_size_acc *= this->outer_size_[idx];
      }
    }

    // Merge strides
    this->strides_[ORDER - 1] = 1;
    for (TensorIdx idx = ORDER - 1; idx > start_idx; --idx)
      this->strides_[idx - 1] = this->outer_size_[idx] * this->strides_[idx];
  }

  // Fill the unused part
  for (TensorIdx idx = 0; idx < start_idx; ++idx) {
    this->size_[idx] = 1;
    this->outer_size_[idx] = 1;
  }
  std::fill(this->offsets_, this->offsets_ + start_idx, 0);
  std::fill(this->strides_, this->strides_ + start_idx, 0);
}


/*
 * Implementation for class ParamTrans
 */
template <typename FloatType,
          TensorOrder ORDER,
          CoefUsageTrans USAGE,
          MemLayout LAYOUT>
ParamTrans<FloatType, ORDER, USAGE, LAYOUT>::ParamTrans(
    const TensorWrapper<FloatType, ORDER, LAYOUT> &input_tensor,
    const TensorWrapper<FloatType, ORDER, LAYOUT> &output_tensor,
    const std::array<TensorOrder, ORDER> &perm,
    DeducedFloatType<FloatType> alpha, DeducedFloatType<FloatType> beta)
    : org_input_tensor(input_tensor),
      org_output_tensor(output_tensor),
      input_tensor(input_tensor),
      output_tensor(output_tensor),
      alpha(alpha), beta(beta),
      input_stride(1), output_stride(1),
      merged_order(ORDER),
      kn_fb(alpha, beta), kn_fv(alpha, beta), kn_fh(alpha, beta),
      kn_fs(alpha, beta), kn_hv(alpha, beta), kn_hh(alpha, beta),
      kn_hs(alpha, beta), kn_sc(alpha, beta) {
  // Initialize perm
  std::copy(perm.begin(), perm.end(), this->perm);

  // Initialize access stride according to memory layout
  if (MemLayout::COL_MAJOR == LAYOUT) {
    for (TensorIdx idx = 0; idx < perm[0]; ++idx)
      this->input_stride *= input_tensor.get_outer_size()[idx];
    for (TensorIdx idx = 0; 0 != perm[idx]; ++idx)
      this->output_stride *= output_tensor.get_outer_size()[idx];
  }
  else {
    for (TensorIdx idx = ORDER - 1; idx > perm[ORDER - 1]; --idx)
      this->input_stride *= input_tensor.get_outer_size()[idx];
    for (TensorIdx idx = ORDER - 1; ORDER - 1 != perm[idx]; --idx)
      this->output_stride *= output_tensor.get_outer_size()[idx];
  }

  // Merge index in tensor wrapper and Initialize merged permutation array
  this->merge_idx_(perm);
}


template <typename FloatType,
          TensorOrder ORDER,
          CoefUsageTrans USAGE,
          MemLayout LAYOUT>
INLINE TensorIdx ParamTrans<FloatType, ORDER, USAGE, LAYOUT>::perm_type() {
  if (MemLayout::COL_MAJOR == LAYOUT) {
    if (0 == this->perm[ORDER - this->merged_order])
      return -1;
    else
      return 1;
  }
  else {
    if (ORDER - 1 == this->perm[ORDER - 1])
      return -1;
    else
      return 1;
  }
}


template <typename FloatType,
          TensorOrder ORDER,
          CoefUsageTrans USAGE,
          MemLayout LAYOUT>
void ParamTrans<FloatType, ORDER, USAGE, LAYOUT>::merge_idx_(
    const std::array<TensorOrder, ORDER> &perm) {
  if (ORDER <= 1)
    return;

  auto input_size = this->input_tensor.get_size();
  auto input_outer_size = this->input_tensor.get_outer_size();
  auto output_size = this->output_tensor.get_size();
  auto output_outer_size = this->output_tensor.get_outer_size();
  std::unordered_set<TensorOrder> input_perm_set, output_perm_set;

  if (MemLayout::COL_MAJOR == LAYOUT) {
    // If column major, then merging will begin from left
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
  }
  else {
    // If row major, then merging will begin from right
    for (TensorIdx idx = ORDER - 2; idx >= 0; --idx) {
      if (perm[idx] + 1 != perm[idx + 1] or
          input_size[perm[idx + 1]] != input_outer_size[perm[idx + 1]] or
          output_size[idx + 1] != output_outer_size[idx + 1]) {
        input_perm_set.insert(perm[idx + 1]);
        output_perm_set.insert(idx + 1);
      }
    }
    input_perm_set.insert(perm[0]);
    output_perm_set.insert(0);
  }

  // Set merged order
  this->merged_order = static_cast<TensorOrder>(input_perm_set.size());
  if (ORDER == this->merged_order)
    return;

  // Update permutation array
  // Create an array for storing sorted keys in input_perm_set,
  TensorOrder sorted_perm_arr[ORDER];
  std::copy(input_perm_set.begin(), input_perm_set.end(), sorted_perm_arr);
  std::sort(sorted_perm_arr, sorted_perm_arr + this->merged_order);

  // Create an unordered map to store the mapping from original order ID to
  // updated order ID.
  std::unordered_map<TensorOrder, TensorOrder> perm_map;
  for (TensorIdx idx = 0; idx < this->merged_order; ++idx)
    perm_map[sorted_perm_arr[idx]] = idx;

  // Update permutation array
  for (TensorIdx idx = ORDER - 1, curr_idx = ORDER - 1; idx >= 0; --idx) {
    if (1 == input_perm_set.count(this->perm[idx])) {
      this->perm[curr_idx] = perm_map[this->perm[idx]];
      --curr_idx;
    }
  }
  // Fill unused part of the permutation array
  std::fill(this->perm, this->perm + ORDER - this->merged_order, 0);

  // Execute merge
  this->input_tensor.merge_idx(input_perm_set);
  this->output_tensor.merge_idx(output_perm_set);
}

#endif // HPTC_PARAM_PARAMETER_TRANS_TCC_
