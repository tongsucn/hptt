#pragma once
#ifndef HPTC_PARAM_PARAMETER_TRANS_H_
#define HPTC_PARAM_PARAMETER_TRANS_H_

#include <array>
#include <unordered_map>
#include <algorithm>

#include <iostream>

#include <hptc/types.h>
#include <hptc/tensor.h>


namespace hptc {

enum class CoefUsage : GenNumType {
  USE_NONE  = 0x0,
  USE_ALPHA = 0x1,
  USE_BETA  = 0x2,
  USE_BOTH  = 0x3
};


template <typename FloatType,
          TensorOrder ORDER,
          MemLayout LAYOUT = MemLayout::COL_MAJOR>
class TensorMergedWrapper : public TensorWrapper<FloatType, ORDER, LAYOUT> {
public:
  TensorMergedWrapper() = delete;
  TensorMergedWrapper(const TensorWrapper<FloatType, ORDER, LAYOUT> &wrapper);

  void merge_idx(const std::unordered_map<TensorOrder, TensorOrder> &merge_map);

  INLINE FloatType &operator[](const TensorIdx * RESTRICT indices);
  INLINE const FloatType &operator[](const TensorIdx * RESTRICT indices) const;
  INLINE FloatType &operator[](TensorIdx **indices);
  INLINE const FloatType &operator[](const TensorIdx **indices) const;

private:
  TensorOrder merged_order_;
};


template <typename FloatType,
          TensorOrder ORDER,
          CoefUsage USAGE,
          MemLayout LAYOUT = MemLayout::COL_MAJOR>
struct ParamTrans {
public:
  ParamTrans(const TensorWrapper<FloatType, ORDER, LAYOUT> &input_tensor,
      const TensorWrapper<FloatType, ORDER, LAYOUT> &output_tensor,
      const std::array<TensorOrder, ORDER> &perm,
      DeducedFloatType<FloatType> alpha, DeducedFloatType<FloatType> beta);

  constexpr static CoefUsage COEF_USAGE = USAGE;
  TensorMergedWrapper<FloatType, ORDER, LAYOUT> input_tensor, output_tensor;
  DeducedFloatType<FloatType> alpha, beta;

  TensorOrder order;
  TensorOrder perm[ORDER];
  TensorIdx input_stride, output_stride;

private:
  void merge_idx_(const std::array<TensorOrder, ORDER> &perm);
};


/*
 * Import implementation
 */
#include "parameter_trans.tcc"

}

#endif // HPTC_PARAM_PARAMETER_TRANS_H_
