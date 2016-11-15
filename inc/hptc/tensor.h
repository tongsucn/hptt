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
  TensorWrapper(TensorWrapper<FloatType> &&wrapper_obj) noexcept;
  TensorWrapper &operator=(const TensorWrapper &wrapper_obj);
  TensorWrapper &operator=(TensorWrapper &&wrapper_obj) noexcept;

  ~TensorWrapper();

  template <typename... Idx>
  FloatType &operator()(Idx... indices);
  template <typename... Ranges>
  TensorWrapper slice(TRI offset, Ranges... range);

  inline const TensorSize &get_size() const;
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


// Import implementation
#include "tensor.tcc"

}


#endif // HPTC_TENSOR_H_
