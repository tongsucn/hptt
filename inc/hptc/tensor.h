#pragma once
#ifndef HPTC_TENSOR_H_
#define HPTC_TENSOR_H_

#include <memory>

#include <hptc/types.h>


namespace hptc {

struct TensorRangeIdx {
  TensorRangeIdx(TensorIdx left, TensorIdx right)
    : left_idx_(left_idx), right_idx_(right_idx) {
  }

  TensorIdx left_idx_;
  TensorIdx right_idx_;
};

using TRI = TensorRangeIdx;

class TensorSize {
public:
  TensorSize();
  TensorSize(TensorDim dim, ...);
  TensorSize(const TensorSize &);
  TensorSize(TensorSize &&);
  TensorSize &operator=(const TensorSize &);
  TensorSize &operator=(TensorSize &&);
  bool operator==(const TensorSize &);

private:
  TensorDim dim_;
  std::shared_ptr<TensorIdx> size_;
  std::shared_ptr<TensorIdx> outer_size_;
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
  TensorSize getDim();

private:
  FloatType *raw_data_;
  TensorSize size_;
};

}


#endif // HPTC_TENSOR_H_
