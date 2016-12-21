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
    : TensorWrapper<FloatType, ORDER, LAYOUT>(wrapper) {
}


template <typename FloatType,
          TensorOrder ORDER,
          MemLayout LAYOUT>
void TensorMergedWrapper<FloatType, ORDER, LAYOUT>::merge_idx(
    const std::unordered_map<TensorOrder, TensorOrder> &merge_map) {
}


template <typename FloatType,
          TensorOrder ORDER,
          MemLayout LAYOUT>
INLINE FloatType &TensorMergedWrapper<FloatType, ORDER, LAYOUT>::operator[](
    const TensorIdx * RESTRICT indices) {
  TensorIdx abs_offset = 0;
  for (TensorIdx idx = 0; idx < this->merged_order_; ++idx)
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
  for (TensorIdx idx = 0; idx < this->merged_order_; ++idx)
    abs_offset += (this->offsets_[idx] + indices[idx]) * this->strides_[idx];
  return this->raw_data_[abs_offset];
}


template <typename FloatType,
          TensorOrder ORDER,
          MemLayout LAYOUT>
INLINE FloatType &TensorMergedWrapper<FloatType, ORDER, LAYOUT>::operator[](
    TensorIdx **indices) {
  TensorIdx abs_offset = 0;
  for (TensorIdx idx = 0; idx < this->merged_order_; ++idx)
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
  for (TensorIdx idx = 0; idx < this->merged_order_; ++idx)
    abs_offset += (this->offsets_[idx] + *indices[idx]) * this->strides_[idx];
  return this->raw_data_[abs_offset];
}


/*
 * Implementation for class ParamTrans
 */
template <typename FloatType,
          TensorOrder ORDER,
          CoefUsage USAGE,
          MemLayout LAYOUT>
ParamTrans<FloatType, ORDER, USAGE, LAYOUT>::ParamTrans(
    const TensorWrapper<FloatType, ORDER, LAYOUT> &input_tensor,
    const TensorWrapper<FloatType, ORDER, LAYOUT> &output_tensor,
    const std::array<TensorOrder, ORDER> &perm,
    DeducedFloatType<FloatType> alpha, DeducedFloatType<FloatType> beta)
    : input_tensor(input_tensor),
      output_tensor(output_tensor),
      alpha(alpha), beta(beta),
      input_stride(1), output_stride(1) {
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
          CoefUsage USAGE,
          MemLayout LAYOUT>
void ParamTrans<FloatType, ORDER, USAGE, LAYOUT>::merge_idx_(
    const std::array<TensorOrder, ORDER> &perm) {
  // TODO: deal with 0-/1-ORDER

  auto input_size = this->input_tensor.get_size();
  auto input_outer_size = this->input_tensor.get_outer_size();
  auto output_size = this->output_tensor.get_size();
  auto output_outer_size = this->output_tensor.get_outer_size();
  std::unordered_map<TensorOrder, TensorOrder> input_perm_map, output_perm_map;

  if (MemLayout::COL_MAJOR == LAYOUT) {
    // If column major, then merge begins from left
    // Create permutation map
    for (TensorOrder idx = 1; idx < ORDER; ++idx) {
      // If current dim ID minus one does not equal to previous dim ID, or the
      // previous dim size does not equal to the outer size, then push
      // previous ID into map.
      if (perm[idx] - 1 != perm[idx - 1] or
          input_size[perm[idx - 1]] != input_outer_size[perm[idx - 1]] or
          output_size[idx - 1] != output_outer_size[idx - 1]) {
        input_perm_map[perm[idx - 1]] = 0;
        output_perm_map[idx - 1] = 0;
      }
    }
    input_perm_map[perm[ORDER - 1]] = 0;
    input_perm_map[ORDER - 1] = 0;
  }
  else {
    // If row major, then merge begins from right
    for (TensorIdx idx = static_cast<TensorIdx>(ORDER - 2); idx >= 0; --idx) {
      if (perm[idx] + 1 != perm[idx + 1] or
          input_size[perm[idx + 1]] != input_outer_size[perm[idx + 1]] or
          output_size[idx + 1] != output_outer_size[idx + 1]) {
        input_perm_map[perm[idx + 1]] = 0;
        output_perm_map[idx + 1] = 0;
      }
    }
    input_perm_map[perm[0]] = 0;
    output_perm_map[0] = 0;
  }

  // Set merged order
  this->order = static_cast<TensorOrder>(input_perm_map.size());
  if (ORDER == this->order)
    return;

  // Execute merge
  this->input_tensor.merge_idx(input_perm_map);
  this->output_tensor.merge_idx(output_perm_map);
}

#endif // HPTC_PARAM_PARAMETER_TRANS_TCC_
