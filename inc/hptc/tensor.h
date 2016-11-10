#pragma once
#ifndef HPTC_TENSOR_H_
#define HPTC_TENSOR_H_

#include <memory>
#include <utility>
#include <initializer_list>

#include <hptc/types.h>


namespace hptc {

struct TensorRangeIdx {
  TensorRangeIdx(TensorIdx left, TensorIdx right)
    : left_idx(left_idx),
      right_idx(right_idx) {
  }

  TensorIdx left_idx;
  TensorIdx right_idx;
};

using TRI = TensorRangeIdx;


class TensorSize {
public:
  TensorSize();
  TensorSize(TensorDim dim);
  TensorSize(std::initializer_list<TensorIdx> size);

  TensorSize(const TensorSize &size_obj);
  TensorSize(TensorSize &&size_obj) noexcept;
  TensorSize &operator=(const TensorSize &size_obj);
  TensorSize &operator=(TensorSize &&size_obj) noexcept;

  ~TensorSize();

  bool operator==(const TensorSize &size_obj);

  inline TensorIdx &operator[](TensorIdx idx);
  inline TensorDim get_dim() const;

private:
  TensorDim dim_;
  TensorIdx *size_;
};


template <typename FloatType>
class TensorWrapper {
public:
  TensorWrapper(const TensorSize &size_obj, FloatType *raw_data);
  TensorWrapper(const TensorSize &size_obj, const TensorSize &outer_size_obj,
      TensorIdx data_offset, FloatType *raw_data);

  TensorWrapper(const TensorWrapper<FloatType> &wrapper_obj);
  TensorWrapper(TensorWrapper<FloatType> &&wrapper_obj);
  TensorWrapper &operator=(const TensorWrapper &wrapper_obj);
  TensorWrapper &operator=(TensorWrapper &&wrapper_obj);

  ~TensorWrapper();

  template <typename... Idx>
  FloatType &operator()(Idx... indices);
  template <typename... Ranges>
  TensorWrapper slice(TRI offset, Ranges... range);

  inline TensorSize get_size() const;
  inline TensorSize get_outer_size() const;
  inline FloatType *get_data();
  inline const FloatType *get_data() const;

private:
  TensorSize size_;
  TensorSize outer_size_;
  TensorIdx data_offset_;
  FloatType *raw_data_;
  TensorIdx *dim_offset_;

  inline void init_dim_offset_();

  template <typename... Idx>
  inline FloatType &get_element_(TensorDim curr_dim, TensorIdx curr_idx,
      TensorIdx next_idx, Idx... idx);
  inline FloatType &get_element_(TensorDim curr_dim, TensorIdx curr_idx);

  template <typename... Ranges>
  inline TensorIdx get_sub_offset_(TensorDim curr_dim, TensorIdx curr_offset,
      TensorSize &size_obj, TRI curr_range, Ranges... range);
  inline TensorIdx get_sub_offset_(TensorDim curr_dim, TensorIdx curr_offset,
      TensorSize &size_obj);
};


// Member function definitions for TensorWrapper
template <typename FloatType>
TensorWrapper<FloatType>::TensorWrapper(const TensorSize &size_obj,
    FloatType *raw_data)
    : size_(size_obj),
      outer_size_(size_obj),
      data_offset_(0),
      raw_data_(raw_data),
      dim_offset_(new TensorIdx [size_obj.get_dim()]) {
  this->init_dim_offset_();
}


template <typename FloatType>
TensorWrapper<FloatType>::TensorWrapper(const TensorSize &size_obj,
    const TensorSize &outer_size_obj, TensorIdx data_offset,
    FloatType *raw_data)
    : size_(size_obj),
      outer_size_(outer_size_obj),
      data_offset_(data_offset),
      raw_data_(raw_data),
      dim_offset_(new TensorIdx [size_obj.get_dim()]) {
  this->init_dim_offset_();
}


template <typename FloatType>
TensorWrapper<FloatType>::TensorWrapper(
    const TensorWrapper<FloatType> &wrapper_obj)
    : size_(wrapper_obj.size_),
      outer_size_(wrapper_obj.outer_size_),
      data_offset_(wrapper_obj.data_offset_),
      raw_data_(wrapper_obj.raw_data_),
      dim_offset_(new TensorIdx [wrapper_obj.size_.get_dim()]) {
  this->init_dim_offset_();
}


  template <typename FloatType>
TensorWrapper<FloatType>::TensorWrapper(TensorWrapper<FloatType> &&wrapper_obj)
    : size_(std::move(wrapper_obj.size_)),
      outer_size_(std::move(wrapper_obj.outer_size_)),
      data_offset_(wrapper_obj.data_offset_),
      raw_data_(wrapper_obj.raw_data_),
      dim_offset_(wrapper_obj.dim_offset_) {
  wrapper_obj.dim_offset_ = nullptr;
}


template <typename FloatType>
TensorWrapper<FloatType>::~TensorWrapper() {
  delete [] this->dim_offset_;
}


template <typename FloatType>
TensorWrapper &TensorWrapper<FloatType>::operator=(
    const TensorWrapper &wrapper_obj) {
  this->size_ = wrapper_obj.size_;
  this->outer_size_ = wrapper_obj.outer_size_;
  this->data_offset_ = wrapper_obj.data_offset_;
  this->raw_data_ = wrapper_obj.raw_data_;
  this->dim_offset_ = new TensorIdx [this->size_.get_dim()];
  this->init_dim_offset_();
}


template <typename FloatType>
TensorWrapper &TensorWrapper<FloatType>::operator=(
    TensorWrapper &&wrapper_obj) {
  this->size_ = std::move(wrapper_obj.size_);
  this->outer_size_ = std::move(wrapper_obj.outer_size_);
  this->data_offset_ = wrapper_obj.data_offset_;
  this->raw_data_ = wrapper_obj.raw_data_;
  this->dim_offset_ = wrapper_obj.dim_offset_;
  wrapper_obj.dim_offset_ = nullptr;
}


template <typename FloatType>
template <typename... Idx>
FloatType &TensorWrapper<FloatType>::operator()(Idx... indices) {
  return this->get_element(0, 0, indices...);
}


template <typename FloatType>
template <typename... Ranges>
TensorWrapper<FloatType> TensorWrapper<FloatType>::slice(Ranges... range) {
  TensorSize size_obj(this->size_.get_dim());
  TensorIdx offset = this->data_offset_
      + this->get_sub_offset_(0, 0, size_obj, range...);
  return TensorWrapper<FloatType>(size_obj, this->outer_size_, offset,
      this->raw_data_);
}


template <typename FloatType>
inline TensorSize TensorWrapper<FloatType>::get_size() const {
  return this->size_;
}


template <typename FloatType>
inline TensorSize TensorWrapper<FloatType>::get_outer_size() const {
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
inline void TensorWrapper<FloatType>::init_dim_offset_() {
  TensorDim dim_upper = this->size_.get_dim() - 1;
  this->dim_offset_[0] = 1;
  for (uint32_t dim_idx = 0; dim_idx < dim_upper; ++dim_idx)
    this->dim_offset_[dim_idx + 1]
        = this->outer_size_[dim_idx] * this->dim_offset_[dim_idx];
}


template <typename FloatType>
template <typename... Idx>
inline FloatType &TensorWrapper<FloatType>::get_element_(TensorDim curr_dim,
    TensorIdx curr_idx, TensorIdx next_idx, Idx... idx) {
  // Compute current index
  if (next_idx < 0)
    next_idx += this->size_[curr_dim];
  curr_idx += next_idx * this->dim_offset_[curr_dim];

  return this->get_element(curr_dim + 1, curr_idx, idx...);
}


template <typename FloatType>
inline FloatType &TensorWrapper<FloatType>::get_element_(TensorDim curr_dim,
    TensorIdx curr_idx) {
  return this->raw_data_[curr_idx + this->data_offset_];
}


template <typename... Ranges>
inline TensorIdx TensorWrapper<FloatType>::get_sub_offset_(TensorDim curr_dim,
    TensorIdx curr_offset, TensorSize &size_obj, TRI curr_range,
    Ranges... range) {
  // Compute size for current dimension
  if (curr_range.right_idx < 0)
    curr_range.right_idx += this->size_[curr_dim];
  if (curr_range.left_idx < 0)
    curr_range.left_idx += this->size_[curr_dim];
  size_obj[curr_dim] = curr_range.right_idx - curr_range.left_idx + 1;

  // Compute current offset
  curr_offset += curr_range.left_idx * this->dim_offset_[curr_dim];

  return this->get_sub_offset_(curr_dim + 1, curr_offset, size_obj, range...);
}


inline TensorIdx TensorWrapper<FloatType>::get_sub_offset_(TensorDim curr_dim,
    TensorIdx curr_offset, TensorSize &size_obj) {
  return curr_offset;
}

}


#endif // HPTC_TENSOR_H_
