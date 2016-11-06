#pragma once
#ifndef HPTC_TENSOR_H_
#define HPTC_TENSOR_H_

#include <memory>
#include <initializer_list>

#include <hptc/types.h>


namespace hptc {

struct TensorRangeIdx {
  TensorRangeIdx(TensorIdx left, TensorIdx right)
    : left_idx(left_idx), right_idx(right_idx) {
  }

  TensorIdx left_idx;
  TensorIdx right_idx;
};

using TRI = TensorRangeIdx;


template<size_t dim>
class TensorSize {
public:
  TensorSize();
  TensorSize(std::initializer_list<TensorIdx> size,
      std::initializer_list<TensorIdx> outer_size = {});
  TensorSize(const TensorSize &size_obj);
  TensorSize(TensorSize &&size_obj) noexcept;

  TensorSize &operator=(const TensorSize &size_obj);
  TensorSize &operator=(TensorSize &&size_obj) noexcept;
  bool operator==(const TensorSize &size_obj);

  TensorIdx &operator[](TensorIdx idx) {
    return this->size_[idx];
  }

  TensorDim GetDim() const {
    return this->dim_;
  }

private:
  TensorDim dim_;
  TensorIdx *size_;
  TensorIdx *outer_size_;
};


template <typename FloatType>
class TensorWrapper {
public:
  TensorWrapper();
  TensorWrapper(const TensorSize &);
  TensorWrapper(const TensorWrapper<FloatType> &);
  TensorWrapper(Tensor &&);
  ~TensorWrapper();

  TensorWrapper &operator=(const TensorWrapper &);
  TensorWrapper &operator=(TensorWrapper &&);
  FloatType operator()(...);

  TensorWrapper Slice(...);
  TensorSize GetSize();
  TensorSize GetDim();

private:
  FloatType *raw_data_;
  TensorSize size_;
};

TensorWrapper::TensorWrapper() {
  ;
}

}


#endif // HPTC_TENSOR_H_
