#pragma once
#ifndef HPTC_TENSOR_TCC_
#define HPTC_TENSOR_TCC_

/*
 * Implementation for class TensorSize
 */
inline TensorIdx &TensorSize::operator[](TensorDim dim_idx) {
  return this->size_[dim_idx];
}


inline const TensorIdx &TensorSize::operator[](TensorDim dim_idx) const {
  return this->size_[dim_idx];
}


inline TensorDim TensorSize::get_dim() const {
  return this->dim_;
}


inline const TensorIdx *TensorSize::shape() const {
  return this->size_;
}


/*
 * Implementation for class TensorWrapper
 */
template <typename FloatType>
TensorWrapper<FloatType>::TensorWrapper()
    : size_(),
      outer_size_(),
      raw_data_(nullptr),
      dim_offset_(nullptr),
      dim_stride_(nullptr) {
}


template <typename FloatType>
TensorWrapper<FloatType>::TensorWrapper(const TensorSize &size_obj,
    FloatType *raw_data)
    : size_(size_obj),
      outer_size_(size_obj),
      raw_data_(0 == size_obj.get_dim() ? nullptr : raw_data),
      dim_offset_(
        0 == size_obj.get_dim() ? nullptr : new TensorIdx [size_obj.get_dim()]),
      dim_stride_(0 == size_obj.get_dim() ?
        nullptr : new TensorIdx [size_obj.get_dim()]) {
  this->init_offset_(std::vector<TensorIdx>());
}


template <typename FloatType>
TensorWrapper<FloatType>::TensorWrapper(const TensorSize &size_obj,
    const TensorSize &outer_size_obj, const std::vector<TensorIdx> &dim_offset,
    FloatType *raw_data)
    : size_(size_obj),
      outer_size_(outer_size_obj),
      raw_data_(0 == size_obj.get_dim() ? nullptr : raw_data),
      dim_offset_(0 == size_obj.get_dim() ?
        nullptr : new TensorIdx [size_obj.get_dim()]),
      dim_stride_(0 == size_obj.get_dim() ?
        nullptr : new TensorIdx [size_obj.get_dim()]) {
  if (this->size_ == this->outer_size_)
    this->init_offset_(std::vector<TensorIdx>());
  else
    this->init_offset_(dim_offset);
}


template <typename FloatType>
TensorWrapper<FloatType>::TensorWrapper(
    const TensorWrapper &wrapper_obj)
    : size_(wrapper_obj.size_),
      outer_size_(wrapper_obj.outer_size_),
      raw_data_(wrapper_obj.raw_data_),
      dim_offset_(0 == size_.get_dim() ?
        nullptr : new TensorIdx [size_.get_dim()]),
      dim_stride_(0 == this->size_.get_dim() ?
        nullptr : new TensorIdx [this->size_.get_dim()]) {
  TensorDim dim = this->size_.get_dim();
  if (0 == dim)
    return;
  std::copy(wrapper_obj.dim_offset_, wrapper_obj.dim_offset_ + dim,
    this->dim_offset_);
  std::copy(wrapper_obj.dim_stride_, wrapper_obj.dim_stride_ + dim,
    this->dim_stride_);
}


  template <typename FloatType>
TensorWrapper<FloatType>::TensorWrapper(
    TensorWrapper &&wrapper_obj) noexcept
    : size_(std::move(wrapper_obj.size_)),
      outer_size_(std::move(wrapper_obj.outer_size_)),
      raw_data_(wrapper_obj.raw_data_),
      dim_offset_(wrapper_obj.dim_offset_),
      dim_stride_(wrapper_obj.dim_stride_) {
  wrapper_obj.dim_offset_ = nullptr;
  wrapper_obj.dim_stride_ = nullptr;
}


template <typename FloatType>
TensorWrapper<FloatType>::~TensorWrapper() {
  delete [] this->dim_offset_;
  delete [] this->dim_stride_;
}


template <typename FloatType>
TensorWrapper<FloatType> &TensorWrapper<FloatType>::operator=(
    const TensorWrapper &wrapper_obj) {
  this->size_ = wrapper_obj.size_;
  TensorDim dim = this->size_.get_dim();
  this->outer_size_ = wrapper_obj.outer_size_;
  this->raw_data_ = wrapper_obj.raw_data_;
  if (0 == dim) {
    this->dim_offset_ = nullptr;
    this->dim_stride_ = nullptr;
  }
  else {
    this->dim_offset_ = new TensorIdx[dim];
    this->dim_stride_ = new TensorIdx[dim];
    std::copy(wrapper_obj.dim_offset_, wrapper_obj.dim_offset_ + dim,
      this->dim_offset_);
    std::copy(wrapper_obj.dim_stride_, wrapper_obj.dim_stride_ + dim,
      this->dim_stride_);
  }
  return *this;
}


template <typename FloatType>
TensorWrapper<FloatType> &TensorWrapper<FloatType>::operator=(
    TensorWrapper &&wrapper_obj) noexcept {
  this->size_ = std::move(wrapper_obj.size_);
  this->outer_size_ = std::move(wrapper_obj.outer_size_);
  this->raw_data_ = wrapper_obj.raw_data_;
  this->dim_offset_ = wrapper_obj.dim_offset_;
  this->dim_stride_ = wrapper_obj.dim_stride_;
  wrapper_obj.dim_offset_ = nullptr;
  wrapper_obj.dim_stride_ = nullptr;
  return *this;
}


template <typename FloatType>
template <typename... Idx>
inline FloatType &TensorWrapper<FloatType>::operator()(Idx... indices) {
  return this->get_element_(0, 0, indices...);
}


template <typename FloatType>
FloatType &TensorWrapper<FloatType>::operator[](
    const std::vector<TensorIdx> &indices) {
  return this->get_element_vec_(indices.begin(), indices.end());
}


template <typename FloatType>
const FloatType &TensorWrapper<FloatType>::operator[](
    const std::vector<TensorIdx> &indices) const {
  return this->get_element_vec_(indices.begin(), indices.end());
}


template <typename FloatType>
FloatType &TensorWrapper<FloatType>::operator[](const TensorIdx *indices) {
  return this->get_element_vec_(indices, indices + this->size_.get_dim());
}


template <typename FloatType>
const FloatType &TensorWrapper<FloatType>::operator[](
    const TensorIdx *indices) const {
  return this->get_element_vec_(indices, indices + this->size_.get_dim());
}


template <typename FloatType>
FloatType &TensorWrapper<FloatType>::operator[](TensorIdx **indices) {
  TensorIdx **begin = indices, **end = indices + this->size_.get_dim();
  TensorIdx abs_offset = 0;
  TensorIdx *dim_stride_ptr = this->dim_stride_;
  while (begin != end) {
    TensorDim dim_idx = std::distance(this->dim_stride_, dim_stride_ptr);
    TensorIdx next_idx = **begin + this->dim_offset_[dim_idx];
    next_idx += **begin < 0 ? this->size_[dim_idx] : 0;
    abs_offset += (*dim_stride_ptr) * next_idx;
    ++begin, ++dim_stride_ptr;
  }

  return *(this->raw_data_ + abs_offset);
}


template <typename FloatType>
const FloatType &TensorWrapper<FloatType>::operator[](
    TensorIdx **indices) const {
  TensorIdx **begin = indices, **end = indices + this->size_.get_dim();
  TensorIdx abs_offset = 0;
  TensorIdx *dim_stride_ptr = this->dim_stride_;
  while (begin != end) {
    TensorDim dim_idx = std::distance(this->dim_stride_, dim_stride_ptr);
    TensorIdx next_idx = **begin + this->dim_offset_[dim_idx];
    next_idx += **begin < 0 ? this->size_[dim_idx] : 0;
    abs_offset += (*dim_stride_ptr) * next_idx;
    ++begin, ++dim_stride_ptr;
  }

  return *(this->raw_data_ + abs_offset);
}


template <typename FloatType>
template <typename... Ranges>
TensorWrapper<FloatType> TensorWrapper<FloatType>::slice(TRI range,
    Ranges... rest) {
  vector<TensorIdx> new_sizes(this->size_.get_dim());
  vector<TensorIdx> new_dim_offset(this->size_.get_dim());
  std::copy(this->dim_offset_, this->dim_offset_ + this->size_.get_dim(),
    new_dim_offset.begin());

  auto sizes_iter = new_sizes.begin();
  auto dim_offset_iter = new_dim_offset.begin();
  this->get_sliced(sizes_iter, dim_offset_iter, range, rest...);
  TensorSize size_obj(new_sizes);

  return TensorWrapper<FloatType>(size_obj, this->outer_size_, new_dim_offset,
    this->raw_data_);
}


template <typename FloatType>
TensorWrapper<FloatType> TensorWrapper<FloatType>::slice(
    const std::vector<TRI> &ranges) {
  return this->get_sliced(ranges.begin(), ranges.end());
}


template <typename FloatType>
TensorWrapper<FloatType> TensorWrapper<FloatType>::slice(const TRI *ranges) {
  return this->get_sliced(ranges, ranges + this->size_.get_dim());
}


template <typename FloatType>
inline const TensorSize &TensorWrapper<FloatType>::get_size() const {
  return this->size_;
}


template <typename FloatType>
inline const TensorSize &TensorWrapper<FloatType>::get_outer_size() const {
  return this->outer_size_;
}


template <typename FloatType>
inline FloatType *TensorWrapper<FloatType>::get_data() {
  return this->raw_data_;
}


template <typename FloatType>
inline const FloatType *TensorWrapper<FloatType>::get_data() const {
  return this->raw_data_;
}


template <typename FloatType>
inline void TensorWrapper<FloatType>::init_offset_(
    const std::vector<TensorIdx> &dim_offset) {
  TensorDim dim = this->size_.get_dim();
  if (0 == this->size_.get_dim())
    return;

  // Initialize dim_offset_
  if (0 == dim_offset.size())
    std::fill(this->dim_offset_, this->dim_offset_ + dim, 0);
  else
    std::copy(dim_offset.begin(), dim_offset.end(), this->dim_offset_);

  // Initialize dim_stride_
  this->dim_stride_[dim - 1] = 1;
  for (TensorDim dim_idx = dim - 1; dim_idx > 0; --dim_idx)
    this->dim_stride_[dim_idx - 1]
      = this->outer_size_[dim_idx] * this->dim_stride_[dim_idx];
}


template <typename FloatType>
template <typename... Idx>
inline FloatType &TensorWrapper<FloatType>::get_element_(TensorDim curr_dim,
    TensorIdx curr_offset, TensorIdx next_idx, Idx... idx) {
  // Compute current index
  if (next_idx < 0)
    next_idx += this->size_[curr_dim];
  next_idx += this->dim_offset_[curr_dim];
  curr_offset += next_idx * this->dim_stride_[curr_dim];

  return this->get_element_(curr_dim + 1, curr_offset, idx...);
}


template <typename FloatType>
inline FloatType &TensorWrapper<FloatType>::get_element_(TensorDim curr_dim,
    TensorIdx curr_offset) {
  return this->raw_data_[curr_offset];
}


template <typename FloatType>
template <typename Iterator>
inline FloatType &TensorWrapper<FloatType>::get_element_vec_(Iterator begin,
    Iterator end) {
  TensorIdx abs_offset = 0;
  TensorIdx *dim_stride_ptr = this->dim_stride_;
  while (begin != end) {
    TensorDim dim_idx = std::distance(this->dim_stride_, dim_stride_ptr);
    TensorIdx next_idx = *begin + this->dim_offset_[dim_idx];
    next_idx += *begin < 0 ? this->size_[dim_idx] : 0;
    abs_offset += (*dim_stride_ptr) * next_idx;
    ++begin, ++dim_stride_ptr;
  }

  return *(this->raw_data_ + abs_offset);
}


template <typename FloatType>
template <typename Iterator>
inline const FloatType &TensorWrapper<FloatType>::get_element_vec_(
    Iterator begin, Iterator end) const {
  TensorIdx abs_offset = 0;
  TensorIdx *dim_stride_ptr = this->dim_stride_;
  while (begin != end) {
    TensorDim dim_idx = std::distance(this->dim_stride_, dim_stride_ptr);
    TensorIdx next_idx = *begin + this->dim_offset_[dim_idx];
    next_idx += *begin < 0 ? this->size_[dim_idx] : 0;
    abs_offset += (*dim_stride_ptr) * next_idx;
    ++begin, ++dim_stride_ptr;
  }

  return *(this->raw_data_ + abs_offset);
}


template <typename FloatType>
template <typename... Ranges>
inline void TensorWrapper<FloatType>::get_sliced(TVecIter &sizes_iter,
    TVecIter &dim_offset_iter, TRI range, Ranges... rest) {
  *size_iter = range.right_idx - range.left_idx;
  *dim_offset_iter += range.left_idx;
  this->get_sliced(++sizes_iter, ++dim_offset_iter, rest...);
  return;
}


template <typename FloatType>
inline void TensorWrapper<FloatType>::get_sliced(TVecIter &sizes_iter,
    TVecIter &dim_offset_iter, TRI range) {
  *size_iter = range.right_idx - range.left_idx;
  *dim_offset_iter += range.left_idx;
  return;
}


template <typename FloatType>
template <typename Iterator>
inline TensorWrapper<FloatType> TensorWrapper<FloatType>::get_sliced(
    Iterator begin, Iterator end) {
  TensorSize size_obj(this->size_.get_dim());
  vector<TensorIdx> new_dim_offset(this->size_.get_dim());
  Iterator ptr = begin;
  while (ptr != end) {
    TensorDim dim_idx = std::distance(begin, ptr);
    size_obj[dim_idx] = ranges[dim_idx].right_idx - ranges[dim_idx].left_idx;
    new_dim_offset[dim_idx]
        = this->dim_offset_[dim_idx] + ranges[dim_idx].left_idx;
    ++ptr;
  }

  return TensorWrapper<FloatType>(size_obj, this->outer_size_, new_dim_offset,
      this->raw_data_);
}

#endif // HPTC_TENSOR_TCC_
