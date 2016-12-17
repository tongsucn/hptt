#pragma once
#ifndef HPTC_TENSOR_TCC_
#define HPTC_TENSOR_TCC_

/*
 * Implementation for class TensorSize
 */
template <TensorOrder ORDER>
TensorSize<ORDER>::TensorSize() {
  std::fill(this->size_, this->size_ + ORDER, 0);
}


template <TensorOrder ORDER>
TensorSize::TensorSize(const std::array<TensorIdx, ORDER> &sizes) {
  std::copy(sizes.begin(), sizes.end(), size_);
}


template <TensorOrder ORDER>
TensorSize<ORDER>::TensorSize(std::initializer_list<TensorIdx> sizes) {
  std::copy(sizes.begin(), sizes.end(), size_);
}


template <TensorOrder ORDER>
bool TensorSize<ORDER>::operator==(const TensorSize<ORDER> &size_obj) const {
  if (not std::equal(this->size_, this->size_ + ORDER, size_obj.size_))
    return false;
  return true;
}


template <TensorOrder ORDER>
INLINE TensorIdx &TensorSize<ORDER>::operator[](TensorOrder order_idx) {
  return this->size_[order_idx];
}


template <TensorOrder ORDER>
INLINE const TensorIdx &TensorSize<ORDER>::operator[](
    TensorOrder order_idx) const {
  return this->size_[order_idx];
}


/*
 * Implementation for class TensorWrapper
 */
template <typename FloatType,
          TensorOrder ORDER,
          MemLayout LAYOUT>
TensorWrapper<FloatType, ORDER, LAYOUT>::TensorWrapper(
    const TensorSize &size_obj, FloatType *raw_data)
    : size_(size_obj),
      outer_size_(size_obj),
      raw_data_(raw_data) {
  this->init_offset_(std::array<TensorIdx, 0>());
}


template <typename FloatType,
          TensorOrder ORDER,
          MemLayout LAYOUT>
TensorWrapper<FloatType, ORDER, LAYOUT>::TensorWrapper(
    const TensorSize &size_obj, const TensorSize &outer_size_obj,
    const std::array<TensorIdx, ORDER> &order_offset, FloatType *raw_data)
    : size_(size_obj),
      outer_size_(outer_size_obj),
      raw_data_(raw_data) {
  if (this->size_ == this->outer_size_)
    this->init_offset_(std::array<TensorIdx, 0>());
  else
    this->init_offset_(order_offset);
}


template <typename FloatType,
          TensorOrder ORDER,
          MemLayout LAYOUT>
template <typename... Idx>
INLINE FloatType &TensorWrapper<FloatType, ORDER, LAYOUT>::operator()(
    Idx... indices) {
  return this->get_element_(0, 0, indices);
}


template <typename FloatType,
          TensorOrder ORDER,
          MemLayout LAYOUT>
INLINE FloatType &TensorWrapper<FloatType, ORDER, LAYOUT>::operator[](
    const TensorIdx *indices) {
  TensorIdx abs_offset = 0;
  for (TensorIdx idx = 0; idx < ORDER; ++idx)
    abs_offset += (this->offsets_[idx] + indices[idx]) * this->strides_[idx];
  return this->raw_data_[abs_offset];
}


template <typename FloatType,
          TensorOrder ORDER,
          MemLayout LAYOUT>
INLINE const FloatType &TensorWrapper<FloatType, ORDER, LAYOUT>::operator[](
    const TensorIdx *indices) const {
  TensorIdx abs_offset = 0;
  for (TensorIdx idx = 0; idx < this->order_; ++idx)
    abs_offset += (this->offsets_[idx] + indices[idx]) * this->strides_[idx];
  return this->raw_data_[abs_offset];
}


template <typename FloatType,
          TensorOrder ORDER,
          MemLayout LAYOUT>
INLINE FloatType &TensorWrapper<FloatType, ORDER, LAYOUT>::operator[](
    TensorIdx **indices) {
  TensorIdx abs_offset = 0;
  for (TensorIdx idx = 0; idx < this->order_; ++idx)
    abs_offset += (this->offsets_[idx] + *indices[idx]) * this->strides_[idx];
  return this->raw_data_[abs_offset];
}


template <typename FloatType,
          TensorOrder ORDER,
          MemLayout LAYOUT>
INLINE const FloatType &TensorWrapper<FloatType, ORDER, LAYOUT>::operator[](
    TensorIdx **indices) const {
  TensorIdx abs_offset = 0;
  for (TensorIdx idx = 0; idx < this->order_; ++idx)
    abs_offset += (this->offsets_[idx] + *indices[idx]) * this->strides_[idx];
  return this->raw_data_[abs_offset];
}


template <typename FloatType,
          TensorOrder ORDER,
          MemLayout LAYOUT>
template <typename... Ranges>
TensorWrapper<FloatType> TensorWrapper<FloatType, ORDER, LAYOUT>::slice(
    TRI range, Ranges... rest) {
}


template <typename FloatType,
          TensorOrder ORDER,
          MemLayout LAYOUT>
TensorWrapper<FloatType> TensorWrapper<FloatType, ORDER, LAYOUT>::slice(
    const std::array<TRI, ORDER> &ranges) {
}


template <typename FloatType,
          TensorOrder ORDER,
          MemLayout LAYOUT>
TensorWrapper<FloatType> TensorWrapper<FloatType, ORDER, LAYOUT>::slice(
    const TRI *ranges) {
}


template <typename FloatType,
          TensorOrder ORDER,
          MemLayout LAYOUT>
INLINE TensorOrder TensorWrapper<FloatType, ORDER, LAYOUT>::get_order() const {
  return ORDER;
}


template <typename FloatType,
          TensorOrder ORDER,
          MemLayout LAYOUT>
INLINE const TensorSize &
TensorWrapper<FloatType, ORDER, LAYOUT>::get_size() const {
  return this->size_;
}


template <typename FloatType,
          TensorOrder ORDER,
          MemLayout LAYOUT>
INLINE const TensorSize &
TensorWrapper<FloatType, ORDER, LAYOUT>::get_outer_size() const {
  return this->outer_size_;
}


template <typename FloatType,
          TensorOrder ORDER,
          MemLayout LAYOUT>
INLINE FloatType *TensorWrapper<FloatType, ORDER, LAYOUT>::get_data() {
  return this->raw_data_;
}


template <typename FloatType,
          TensorOrder ORDER,
          MemLayout LAYOUT>
INLINE const FloatType *
TensorWrapper<FloatType, ORDER, LAYOUT>::get_data() const {
  return this->raw_data_;
}


template <typename FloatType,
          TensorOrder ORDER,
          MemLayout LAYOUT>
INLINE void TensorWrapper<FloatType, ORDER, LAYOUT>::init_offset_(
    const std::array<TensorIdx, ORDER> &order_offset) {
  if (0 == ORDER)
    return;

  // Initialize offsets_
  std::copy(order_offset.begin(), order_offset.end(), this->offsets_);

  // Initialize strides_ according to memory layout
  if (MemLayout::COL_MAJOR == LAYOUT) {
    this->strides_[0] = 1;
    for (TensorOrder order_idx = 0; order_idx < this->order_ - 1; ++order_idx)
      this->strides_[order_idx + 1]
          = this->outer_size_[order_idx] * this->strides_[order_idx];
  }
  else {
    this->strides_[this->order_ - 1] = 1;
    for (TensorOrder order_idx = this->order_ - 1; order_idx > 0; --order_idx)
      this->strides_[order_idx - 1]
          = this->outer_size_[order_idx] * this->strides_[order_idx];
  }
}


template <typename FloatType,
          TensorOrder ORDER,
          MemLayout LAYOUT>
template <typename... Idx>
INLINE FloatType &TensorWrapper<FloatType, ORDER, LAYOUT>::get_element_(
    TensorOrder curr_order, TensorIdx curr_offset, TensorIdx next_idx,
    Idx... idx) {
  // Compute current index
  next_idx += this->offsets_[curr_order];
  curr_offset += next_idx * this->strides_[curr_order];

  return this->get_element_(curr_order + 1, curr_offset, idx...);
}


template <typename FloatType,
          TensorOrder ORDER,
          MemLayout LAYOUT>
INLINE FloatType &TensorWrapper<FloatType, ORDER, LAYOUT>::get_element_(
    TensorOrder curr_order, TensorIdx curr_offset) {
  return this->raw_data_[curr_offset];
}

#endif // HPTC_TENSOR_TCC_
