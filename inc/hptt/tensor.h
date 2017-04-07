#pragma once
#ifndef HPTT_TENSOR_H_
#define HPTT_TENSOR_H_

#include <array>
#include <vector>
#include <algorithm>
#include <initializer_list>

#include <hptt/types.h>
#include <hptt/arch/compat.h>


namespace hptt {

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
          TensorUInt ORDER>
class TensorWrapper {
public:
  using Float = FloatType;
  constexpr static auto TENSOR_ORDER = ORDER;

  TensorWrapper();

  TensorWrapper(const TensorSize<ORDER> &size_obj, const FloatType *raw_data);
  TensorWrapper(const TensorSize<ORDER> &size_obj,
      const TensorSize<ORDER> &outer_size_obj, const FloatType *raw_data);

  TensorWrapper(const TensorWrapper<FloatType, ORDER> &wrapper);

  FloatType &operator[](const TensorIdx * RESTRICT indices);
  const FloatType &operator[](const TensorIdx * RESTRICT indices) const;

  const TensorSize<ORDER> &get_size() const;
  TensorIdx get_size(const TensorUInt order) const;

  const TensorSize<ORDER> &get_outer_size() const;
  TensorIdx get_outer_size(const TensorUInt order) const;

  void reset_data(const Float *new_data);
  FloatType *get_data();
  const FloatType *get_data() const;

protected:
  // Internal function member
  void init_strides_();

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

#endif // HPTT_TENSOR_H_
