#pragma once
#ifndef HPTC_TENSOR_H_
#define HPTC_TENSOR_H_

#include <array>
#include <vector>
#include <algorithm>
#include <initializer_list>

#include <hptc/types.h>
#include <hptc/arch/compat.h>


namespace hptc {

template <TensorUInt ORDER>
class TensorSize {
public:
  TensorSize();
  TensorSize(const std::array<TensorIdx, ORDER> &sizes);
  TensorSize(const std::vector<TensorIdx> &sizes);
  TensorSize(std::initializer_list<TensorIdx> sizes);

  bool operator==(const TensorSize<ORDER> &size_obj) const;

  TensorIdx &operator[](TensorUInt order);
  const TensorIdx &operator[](TensorUInt order) const;

private:
  TensorIdx size_[ORDER];
};


template <typename FloatType,
          TensorUInt ORDER,
          MemLayout LAYOUT = MemLayout::COL_MAJOR>
class TensorWrapper {
public:
  using Float = FloatType;
  constexpr static auto TENSOR_ORDER = ORDER;

  TensorWrapper();

  TensorWrapper(const TensorSize<ORDER> &size_obj, const FloatType *raw_data);
  TensorWrapper(const TensorSize<ORDER> &size_obj,
      const TensorSize<ORDER> &outer_size_obj, const FloatType *raw_data);

  template <MemLayout ACT_MAJOR>
  TensorWrapper(const TensorWrapper<FloatType, ORDER, ACT_MAJOR> &wrapper);

  template <typename... Idx>
  FloatType &operator()(Idx... indices);

  FloatType &operator[](const TensorIdx * RESTRICT indices);
  const FloatType &operator[](const TensorIdx * RESTRICT indices) const;

  const TensorSize<ORDER> &get_size() const;
  const TensorSize<ORDER> &get_outer_size() const;
  FloatType *get_data();
  const FloatType *get_data() const;

protected:
  // Internal function member
  void init_strides_();

  template <typename... Idx>
  FloatType &get_element_(TensorUInt curr_order, TensorIdx abs_offset,
      TensorIdx next_idx, Idx... idx);
  FloatType &get_element_(TensorUInt curr_order, TensorIdx abs_offset);

  // Internal data member
  TensorSize<ORDER> size_;
  TensorSize<ORDER> outer_size_;
  FloatType *raw_data_;
  TensorIdx strides_[ORDER];
};


/*
 * Import implementation
 */
#include "tensor.tcc"

}

#endif // HPTC_TENSOR_H_
