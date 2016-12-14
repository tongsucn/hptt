#pragma once
#ifndef HPTC_TENSOR_H_
#define HPTC_TENSOR_H_

#include <vector>
#include <memory>
#include <utility>
#include <iterator>
#include <initializer_list>

#include <hptc/types.h>


namespace hptc {

struct TensorRangeIdx {
  TensorRangeIdx(TensorIdx left_idx, TensorIdx right_idx)
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
  TensorSize(const std::vector<TensorIdx> &sizes);
  TensorSize(std::initializer_list<TensorIdx> sizes);

  TensorSize(const TensorSize &size_obj);
  TensorSize(TensorSize &&size_obj) noexcept;
  TensorSize &operator=(const TensorSize &size_obj);
  TensorSize &operator=(TensorSize &&size_obj) noexcept;

  ~TensorSize();

  bool operator==(const TensorSize &size_obj) const;

  inline TensorIdx &operator[](TensorDim dim_idx);
  inline const TensorIdx &operator[](TensorDim dim_idx) const;
  inline TensorDim get_dim() const;
  inline const TensorIdx *shape() const;

private:
  TensorDim dim_;
  TensorIdx *size_;
};


template <typename FloatType>
class TensorWrapper {
public:
  TensorWrapper();
  TensorWrapper(const TensorSize &size_obj, FloatType *raw_data);
  TensorWrapper(const TensorSize &size_obj, const TensorSize &outer_size_obj,
    const std::vector<TensorIdx> &dim_offset, FloatType *raw_data);

  TensorWrapper(const TensorWrapper &wrapper_obj);
  TensorWrapper(TensorWrapper &&wrapper_obj) noexcept;
  TensorWrapper &operator=(const TensorWrapper &wrapper_obj);
  TensorWrapper &operator=(TensorWrapper &&wrapper_obj) noexcept;

  ~TensorWrapper();

  template <typename... Idx>
  inline FloatType &operator()(Idx... indices);

  FloatType &operator[](const std::vector<TensorIdx> &indices);
  const FloatType &operator[](const std::vector<TensorIdx> &indices) const;
  FloatType &operator[](const TensorIdx *indices);
  const FloatType &operator[](const TensorIdx *indices) const;
  FloatType &operator[](TensorIdx **indices);
  const FloatType &operator[](TensorIdx **indices) const;

  template <typename... Ranges>
  TensorWrapper<FloatType> slice(TRI range, Ranges... rest);
  TensorWrapper<FloatType> slice(const std::vector<TRI> &ranges);
  TensorWrapper<FloatType> slice(const TRI *ranges);

  inline const TensorSize &get_size() const;
  inline const TensorSize &get_outer_size() const;
  inline FloatType *get_data();
  inline const FloatType *get_data() const;

private:
  TensorSize size_;
  TensorSize outer_size_;
  FloatType *raw_data_;
  TensorIdx *dim_offset_;
  TensorIdx *dim_stride_;

  inline void init_offset_(const std::vector<TensorIdx> &dim_offset);

  template <typename... Idx>
  inline FloatType &get_element_(TensorDim curr_dim, TensorIdx curr_offset,
    TensorIdx next_idx, Idx... idx);
  inline FloatType &get_element_(TensorDim curr_dim, TensorIdx curr_offset);

  template <typename Iterator>
  inline FloatType &get_element_vec_(Iterator begin, Iterator end);
  template <typename Iterator>
  inline const FloatType &get_element_vec_(Iterator begin, Iterator end) const;

  using TVecIter = std::vector<TensorIdx>::iterator;

  template <typename... Ranges>
  inline void get_sliced(TVecIter &sizes_iter, TVecIter &dim_offset_iter,
    TRI range, Ranges... rest);
  inline void get_sliced(TVecIter &sizes_iter, TVecIter &dim_offset_iter,
    TRI range);
  template <typename Iterator>
  inline TensorWrapper<FloatType> get_sliced(Iterator begin, Iterator end);
};


// Import implementation
#include "tensor.tcc"

}

#endif // HPTC_TENSOR_H_
