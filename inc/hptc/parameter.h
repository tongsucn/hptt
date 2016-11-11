#pragma once
#ifndef HPTC_PARAMETER_H_
#define HPTC_PARAMETER_H_

#include <hptc/types.h>
#include <hptc/tensor.h>

namespace hptc {

template <typename FloatType>
struct ParamTrans {
  ParamTrans();

  TensorWrapper<FloatType> &input_tensor;
  TensorWrapper<FloatType> &output_tensor;

  CoefficientType<FloatType> alpha;
  CoefficientType<FloatType> beta;
};

}


#endif // HPTC_PARAMETER_H_
