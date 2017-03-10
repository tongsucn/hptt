#pragma once
#ifndef HPTC_TENSOR_TCC_
#define HPTC_TENSOR_TCC_

/*
 * Implementation for class TensorSize
 */
class TensorSize<0> {
  TensorSize() = delete;
};


class TensorSize<1> {
  TensorSize() = delete;
};


template <TensorOrder ORDER>
TensorSize<ORDER>::TensorSize() {
  std::fill(this->size_, this->size_ + ORDER, 0);
}


template <TensorOrder ORDER>
TensorSize<ORDER>::TensorSize(const std::array<TensorOrder, ORDER> &sizes) {
  std::copy(sizes.begin(), sizes.end(), size_);
}


template <TensorOrder ORDER>
TensorSize<ORDER>::TensorSize(const std::vector<TensorOrder> &sizes) {
  std::copy(sizes.begin(), sizes.end(), size_);
}


template <TensorOrder ORDER>
TensorSize<ORDER>::TensorSize(std::initializer_list<TensorOrder> sizes) {
  std::copy(sizes.begin(), sizes.end(), size_);
}


template <TensorOrder ORDER>
bool TensorSize<ORDER>::operator==(const TensorSize &size_obj) const {
  if (not std::equal(this->size_, this->size_ + ORDER, size_obj.size_))
    return false;
  return true;
}


template <TensorOrder ORDER>
INLINE TensorOrder &TensorSize<ORDER>::operator[](TensorOrder order_idx) {
  return this->size_[order_idx];
}


template <TensorOrder ORDER>
INLINE const TensorOrder &TensorSize<ORDER>::operator[](
    TensorOrder order_idx) const {
  return this->size_[order_idx];
}


/*
 * Implementation for class TensorWrapper
 */
template <typename FloatType,
          TensorOrder ORDER,
          MemLayout LAYOUT>
TensorWrapper<FloatType, ORDER, LAYOUT>::TensorWrapper()
    : size_(),
      outer_size_(),
      raw_data_(nullptr) {
  this->init_offset_();
}


template <typename FloatType,
          TensorOrder ORDER,
          MemLayout LAYOUT>
TensorWrapper<FloatType, ORDER, LAYOUT>::TensorWrapper(
    const TensorSize<ORDER> &size_obj, FloatType *raw_data)
    : size_(size_obj),
      outer_size_(size_obj),
      raw_data_(raw_data) {
  this->init_offset_();
}


template <typename FloatType,
          TensorOrder ORDER,
          MemLayout LAYOUT>
TensorWrapper<FloatType, ORDER, LAYOUT>::TensorWrapper(
    const TensorSize<ORDER> &size_obj, const TensorSize<ORDER> &outer_size_obj,
    const std::array<TensorIdx, ORDER> &order_offset, FloatType *raw_data)
    : size_(size_obj),
      outer_size_(outer_size_obj),
      raw_data_(raw_data) {
  if (this->size_ == this->outer_size_)
    this->init_offset_();
  else
    this->init_offset_(order_offset);
}


template <typename FloatType,
          TensorOrder ORDER,
          MemLayout LAYOUT>
template <MemLayout ACT_MAJOR>
TensorWrapper<FloatType, ORDER, LAYOUT>::TensorWrapper(
    TensorWrapper<FloatType, ORDER, ACT_MAJOR> &wrapper)
    : size_(wrapper.get_size()),
      outer_size_(wrapper.get_outer_size()),
      raw_data_(wrapper.get_data()) {
  std::array<TensorIdx, ORDER> order_offset;
  // Translate if input wrapper has different layout
  if (LAYOUT != ACT_MAJOR) {
    // Reverse size objects
    TensorIdx left_idx = 0, right_idx = ORDER - 1;
    while (left_idx < right_idx) {
      std::swap(this->size_[left_idx], this->size_[right_idx]);
      std::swap(this->outer_size_[left_idx], this->outer_size_[right_idx]);
      ++left_idx, --right_idx;
    }

    // Reverse offsets
    std::reverse_copy(wrapper.get_offset(), wrapper.get_offset() + ORDER,
        order_offset.begin());
  }
  else
    std::copy(wrapper.get_offset(), wrapper.get_offset() + ORDER,
        order_offset.begin());

  // this->strides_ and this->offsets_ are initialized here
  if (this->size_ == this->outer_size_)
    this->init_offset_();
  else
    this->init_offset_(order_offset);
  this->init_offset_();
}


template <typename FloatType,
          TensorOrder ORDER,
          MemLayout LAYOUT>
template <typename... Idx>
INLINE FloatType &TensorWrapper<FloatType, ORDER, LAYOUT>::operator()(
    Idx... indices) {
  return this->get_element_(0, 0, indices...);
}


template <typename FloatType,
          TensorOrder ORDER,
          MemLayout LAYOUT>
INLINE FloatType &TensorWrapper<FloatType, ORDER, LAYOUT>::operator[](
    const TensorIdx * RESTRICT indices) {
  TensorIdx abs_offset = 0;
  for (TensorIdx idx = 0; idx < ORDER; ++idx)
    abs_offset += (this->offsets_[idx] + indices[idx]) * this->strides_[idx];
  return this->raw_data_[abs_offset];
}


template <typename FloatType,
          TensorOrder ORDER,
          MemLayout LAYOUT>
INLINE const FloatType &TensorWrapper<FloatType, ORDER, LAYOUT>::operator[](
    const TensorIdx * RESTRICT indices) const {
  TensorIdx abs_offset = 0;
  for (TensorIdx idx = 0; idx < ORDER; ++idx)
    abs_offset += (this->offsets_[idx] + indices[idx]) * this->strides_[idx];
  return this->raw_data_[abs_offset];
}


template <typename FloatType,
          TensorOrder ORDER,
          MemLayout LAYOUT>
template <typename... Ranges>
TensorWrapper<FloatType, ORDER, LAYOUT>
TensorWrapper<FloatType, ORDER, LAYOUT>::slice(TRI range, Ranges... rest) {
}


template <typename FloatType,
          TensorOrder ORDER,
          MemLayout LAYOUT>
TensorWrapper<FloatType, ORDER, LAYOUT>
TensorWrapper<FloatType, ORDER, LAYOUT>::slice(
    const std::array<TRI, ORDER> &ranges) {
  return *this;
}


template <typename FloatType,
          TensorOrder ORDER,
          MemLayout LAYOUT>
INLINE TensorSize<ORDER> &TensorWrapper<FloatType, ORDER, LAYOUT>::get_size() {
  return this->size_;
}


template <typename FloatType,
          TensorOrder ORDER,
          MemLayout LAYOUT>
INLINE const TensorSize<ORDER> &
TensorWrapper<FloatType, ORDER, LAYOUT>::get_size() const {
  return this->size_;
}


template <typename FloatType,
          TensorOrder ORDER,
          MemLayout LAYOUT>
INLINE TensorSize<ORDER> &
TensorWrapper<FloatType, ORDER, LAYOUT>::get_outer_size() {
  return this->outer_size_;
}


template <typename FloatType,
          TensorOrder ORDER,
          MemLayout LAYOUT>
INLINE const TensorSize<ORDER> &
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
INLINE void
TensorWrapper<FloatType, ORDER, LAYOUT>::set_data(FloatType *new_data) {
  this->raw_data_ = new_data;
}


template <typename FloatType,
          TensorOrder ORDER,
          MemLayout LAYOUT>
INLINE const TensorIdx *TensorWrapper<FloatType, ORDER, LAYOUT>::get_offset(
    ) const {
  return this->offsets_;
}


template <typename FloatType,
          TensorOrder ORDER,
          MemLayout LAYOUT>
INLINE void TensorWrapper<FloatType, ORDER, LAYOUT>::init_offset_() {
  if (0 == ORDER)
    return;

  // Initialize offsets_
  std::fill(this->offsets_, this->offsets_ + ORDER, 0);

  // Initialize strides_ according to memory layout
  if (MemLayout::COL_MAJOR == LAYOUT) {
    this->strides_[0] = 1;
    for (TensorOrder order_idx = 0; order_idx < ORDER - 1; ++order_idx)
      this->strides_[order_idx + 1]
          = this->outer_size_[order_idx] * this->strides_[order_idx];
  }
  else {
    this->strides_[ORDER - 1] = 1;
    for (TensorIdx order_idx = ORDER - 1; order_idx > 0; --order_idx)
      this->strides_[order_idx - 1]
          = this->outer_size_[order_idx] * this->strides_[order_idx];
  }
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
    for (TensorOrder order_idx = 0; order_idx < ORDER - 1; ++order_idx)
      this->strides_[order_idx + 1]
          = this->outer_size_[order_idx] * this->strides_[order_idx];
  }
  else {
    this->strides_[ORDER - 1] = 1;
    for (TensorIdx order_idx = ORDER - 1; order_idx > 0; --order_idx)
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
