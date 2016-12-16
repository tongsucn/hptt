#pragma once
#ifndef HPTC_TENSOR_TCC_
#define HPTC_TENSOR_TCC_

/*
 * Implementation for class TensorSize
 */
INLINE TensorIdx &TensorSize::operator[](TensorDim dim_idx) {
  return this->size_[dim_idx];
}


INLINE const TensorIdx &TensorSize::operator[](TensorDim dim_idx) const {
  return this->size_[dim_idx];
}


INLINE TensorDim TensorSize::get_dim() const {
  return this->dim_;
}


INLINE const TensorIdx *TensorSize::shape() const {
  return this->size_;
}


/*
 * Implementation for class TensorWrapper
 */
template <typename FloatType,
          MemLayout LAYOUT>
TensorWrapper<FloatType>::TensorWrapper()
    : size_(),
      outer_size_(),
      dim_(0),
      raw_data_(nullptr),
      dim_offset_(nullptr),
      dim_stride_(nullptr) {
}


template <typename FloatType,
          MemLayout LAYOUT>
TensorWrapper<FloatType>::TensorWrapper(const TensorSize &size_obj,
    FloatType *raw_data)
    : size_(size_obj),
      outer_size_(size_obj),
      dim_(this->size_.get_dim()),
      raw_data_(0 == this->dim_ ? nullptr : raw_data),
      dim_offset_(0 == this->dim_ ? nullptr : new TensorIdx [this->dim_]),
      dim_stride_(0 == this->dim_ ? nullptr : new TensorIdx [this->dim_]) {
  this->init_offset_(std::vector<TensorIdx>());
}


template <typename FloatType,
          MemLayout LAYOUT>
TensorWrapper<FloatType>::TensorWrapper(const TensorSize &size_obj,
    const TensorSize &outer_size_obj, const std::vector<TensorIdx> &dim_offset,
    FloatType *raw_data)
    : size_(size_obj),
      outer_size_(outer_size_obj),
      dim_(this->size_.get_dim()),
      raw_data_(0 == this->dim_ ? nullptr : raw_data),
      dim_offset_(0 == this->dim_ ? nullptr : new TensorIdx [this->dim_]),
      dim_stride_(0 == this->dim_ ? nullptr : new TensorIdx [this->dim_]) {
  if (this->size_ == this->outer_size_)
    this->init_offset_(std::vector<TensorIdx>());
  else
    this->init_offset_(dim_offset);
}


template <typename FloatType,
          MemLayout LAYOUT>
TensorWrapper<FloatType>::TensorWrapper(const TensorWrapper &wrapper_obj)
    : size_(wrapper_obj.size_),
      outer_size_(wrapper_obj.outer_size_),
      dim_(wrapper_obj.dim_),
      raw_data_(wrapper_obj.raw_data_),
      dim_offset_(0 == this->dim_ ? nullptr : new TensorIdx [this->dim_]),
      dim_stride_(0 == this->dim_ ? nullptr : new TensorIdx [this->dim_]) {
  if (0 == this->dim_)
    return;
  std::copy(wrapper_obj.dim_offset_, wrapper_obj.dim_offset_ + this->dim_,
      this->dim_offset_);
  std::copy(wrapper_obj.dim_stride_, wrapper_obj.dim_stride_ + this->dim_,
      this->dim_stride_);
}


template <typename FloatType,
          MemLayout LAYOUT>
TensorWrapper<FloatType>::TensorWrapper(TensorWrapper &&wrapper_obj) noexcept
    : size_(std::move(wrapper_obj.size_)),
      outer_size_(std::move(wrapper_obj.outer_size_)),
      dim_(wrapper_obj.dim_),
      raw_data_(wrapper_obj.raw_data_),
      dim_offset_(wrapper_obj.dim_offset_),
      dim_stride_(wrapper_obj.dim_stride_) {
  wrapper_obj.dim_offset_ = nullptr;
  wrapper_obj.dim_stride_ = nullptr;
}


template <typename FloatType,
          MemLayout LAYOUT>
TensorWrapper<FloatType>::~TensorWrapper() {
  delete [] this->dim_offset_;
  delete [] this->dim_stride_;
}


template <typename FloatType,
          MemLayout LAYOUT>
TensorWrapper<FloatType> &TensorWrapper<FloatType>::operator=(
    const TensorWrapper &wrapper_obj) {
  this->size_ = wrapper_obj.size_;
  this->outer_size_ = wrapper_obj.outer_size_;
  this->dim_ = wrapper_obj.dim_;
  this->raw_data_ = wrapper_obj.raw_data_;

  if (0 == this->dim_) {
    this->dim_offset_ = nullptr;
    this->dim_stride_ = nullptr;
  }
  else {
    this->dim_offset_ = new TensorIdx[this->dim_];
    this->dim_stride_ = new TensorIdx[this->dim_];
    std::copy(wrapper_obj.dim_offset_, wrapper_obj.dim_offset_ + this->dim_,
        this->dim_offset_);
    std::copy(wrapper_obj.dim_stride_, wrapper_obj.dim_stride_ + this->dim_,
        this->dim_stride_);
  }

  return *this;
}


template <typename FloatType,
          MemLayout LAYOUT>
TensorWrapper<FloatType> &TensorWrapper<FloatType>::operator=(
    TensorWrapper &&wrapper_obj) noexcept {
  this->size_ = std::move(wrapper_obj.size_);
  this->outer_size_ = std::move(wrapper_obj.outer_size_);
  this->dim_ = wrapper_obj.dim_;
  this->raw_data_ = wrapper_obj.raw_data_;

  this->dim_offset_ = wrapper_obj.dim_offset_;
  this->dim_stride_ = wrapper_obj.dim_stride_;
  wrapper_obj.dim_offset_ = nullptr;
  wrapper_obj.dim_stride_ = nullptr;

  return *this;
}


template <typename FloatType,
          MemLayout LAYOUT>
template <typename... Idx>
INLINE FloatType &TensorWrapper<FloatType>::operator()(Idx... indices) {
  return this->get_element_(0, 0, indices...);
}


template <typename FloatType,
          MemLayout LAYOUT>
INLINE FloatType &TensorWrapper<FloatType>::operator[](
    const TensorIdx *indices) {
  TensorIdx abs_offset = 0;
  for (TensorIdx idx = 0; idx < this->dim_; ++idx)
    abs_offset
        += (this->dim_offset_[idx] + indices[idx]) * this->dim_stride_[idx];
  return this->raw_data_[abs_offset];
}


template <typename FloatType,
          MemLayout LAYOUT>
INLINE const FloatType &TensorWrapper<FloatType>::operator[](
    const TensorIdx *indices) const {
  TensorIdx abs_offset = 0;
  for (TensorIdx idx = 0; idx < this->dim_; ++idx)
    abs_offset
        += (this->dim_offset_[idx] + indices[idx]) * this->dim_stride_[idx];
  return this->raw_data_[abs_offset];
}


template <typename FloatType,
          MemLayout LAYOUT>
INLINE FloatType &TensorWrapper<FloatType>::operator[](TensorIdx **indices) {
  TensorIdx abs_offset = 0;
  for (TensorIdx idx = 0; idx < this->dim_; ++idx)
    abs_offset
        += (this->dim_offset_[idx] + *indices[idx]) * this->dim_stride_[idx];
  return this->raw_data_[abs_offset];
}


template <typename FloatType,
          MemLayout LAYOUT>
INLINE const FloatType &TensorWrapper<FloatType>::operator[](
    TensorIdx **indices) const {
  TensorIdx abs_offset = 0;
  for (TensorIdx idx = 0; idx < this->dim_; ++idx)
    abs_offset
        += (this->dim_offset_[idx] + *indices[idx]) * this->dim_stride_[idx];
  return this->raw_data_[abs_offset];
}


template <typename FloatType,
          MemLayout LAYOUT>
template <typename... Ranges>
TensorWrapper<FloatType> TensorWrapper<FloatType>::slice(TRI range,
    Ranges... rest) {
}


template <typename FloatType,
          MemLayout LAYOUT>
TensorWrapper<FloatType> TensorWrapper<FloatType>::slice(
    const std::vector<TRI> &ranges) {
}


template <typename FloatType,
          MemLayout LAYOUT>
TensorWrapper<FloatType> TensorWrapper<FloatType>::slice(const TRI *ranges) {
}


template <typename FloatType,
          MemLayout LAYOUT>
INLINE TensorDim TensorWrapper<FloatType>::get_dim() const {
  return this->dim_;
}


template <typename FloatType,
          MemLayout LAYOUT>
INLINE const TensorSize &TensorWrapper<FloatType>::get_size() const {
  return this->size_;
}


template <typename FloatType,
          MemLayout LAYOUT>
INLINE const TensorSize &TensorWrapper<FloatType>::get_outer_size() const {
  return this->outer_size_;
}


template <typename FloatType,
          MemLayout LAYOUT>
INLINE FloatType *TensorWrapper<FloatType>::get_data() {
  return this->raw_data_;
}


template <typename FloatType,
          MemLayout LAYOUT>
INLINE const FloatType *TensorWrapper<FloatType>::get_data() const {
  return this->raw_data_;
}


template <typename FloatType,
          MemLayout LAYOUT>
INLINE void TensorWrapper<FloatType>::init_offset_(
    const std::vector<TensorIdx> &dim_offset) {
  if (0 == this->dim_)
    return;

  // Initialize dim_offset_
  if (0 == dim_offset.size())
    std::fill(this->dim_offset_, this->dim_offset_ + this->dim_, 0);
  else
    std::copy(dim_offset.begin(), dim_offset.end(), this->dim_offset_);

  // Initialize dim_stride_ according to memory layout
  if (MemLayout::COL_MAJOR == LAYOUT) {
    this->dim_stride_[0] = 1;
    for (TensorDim dim_idx = 0; dim_idx < this->dim_ - 1; ++dim_idx)
      this->dim_stride_[dim_idx + 1]
          = this->outer_size_[dim_idx] * this->dim_stride_[dim_idx];
  }
  else {
    this->dim_stride_[this->dim_ - 1] = 1;
    for (TensorDim dim_idx = this->dim_ - 1; dim_idx > 0; --dim_idx)
      this->dim_stride_[dim_idx - 1]
          = this->outer_size_[dim_idx] * this->dim_stride_[dim_idx];
  }
}


template <typename FloatType,
          MemLayout LAYOUT>
template <typename... Idx>
INLINE FloatType &TensorWrapper<FloatType>::get_element_(TensorDim curr_dim,
    TensorIdx curr_offset, TensorIdx next_idx, Idx... idx) {
  // Compute current index
  next_idx += this->dim_offset_[curr_dim];
  curr_offset += next_idx * this->dim_stride_[curr_dim];

  return this->get_element_(curr_dim + 1, curr_offset, idx...);
}


template <typename FloatType,
          MemLayout LAYOUT>
INLINE FloatType &TensorWrapper<FloatType>::get_element_(TensorDim curr_dim,
    TensorIdx curr_offset) {
  return this->raw_data_[curr_offset];
}

#endif // HPTC_TENSOR_TCC_
