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

  INLINE TensorIdx &operator[](TensorDim dim_idx);
  INLINE const TensorIdx &operator[](TensorDim dim_idx) const;
  INLINE TensorDim get_dim() const;
  INLINE const TensorIdx *shape() const;

private:
  TensorDim dim_;
  TensorIdx *size_;
};


template <typename FloatType,
          MemLayout LAYOUT = MemLayout::COL_MAJOR>
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
  INLINE FloatType &operator()(Idx... indices);

  INLINE FloatType &operator[](const TensorIdx *indices);
  INLINE const FloatType &operator[](const TensorIdx *indices) const;
  INLINE FloatType &operator[](TensorIdx **indices);
  INLINE const FloatType &operator[](TensorIdx **indices) const;

  template <typename... Ranges>
  TensorWrapper<FloatType> slice(TRI range, Ranges... rest);
  TensorWrapper<FloatType> slice(const std::vector<TRI> &ranges);
  TensorWrapper<FloatType> slice(const TRI *ranges);

  INLINE TensorDim get_dim() const;
  INLINE const TensorSize &get_size() const;
  INLINE const TensorSize &get_outer_size() const;
  INLINE FloatType *get_data();
  INLINE const FloatType *get_data() const;

private:
  // Internal function member
  INLINE void init_offset_(const std::vector<TensorIdx> &dim_offset);

  template <typename... Idx>
  INLINE FloatType &get_element_(TensorDim curr_dim, TensorIdx curr_offset,
    TensorIdx next_idx, Idx... idx);
  INLINE FloatType &get_element_(TensorDim curr_dim, TensorIdx curr_offset);

  // Internal data member
  TensorSize size_;
  TensorSize outer_size_;
  TensorDim dim_;
  FloatType *raw_data_;
  TensorIdx *dim_offset_;
  TensorIdx *dim_stride_;
};


// Import implementation
#include "tensor.tcc"

}

#endif // HPTC_TENSOR_H_
