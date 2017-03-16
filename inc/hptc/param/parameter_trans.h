#pragma once
#ifndef HPTC_PARAM_PARAMETER_TRANS_H_
#define HPTC_PARAM_PARAMETER_TRANS_H_

#include <array>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

#include <hptc/types.h>
#include <hptc/tensor.h>
#include <hptc/config/config_trans.h>
#include <hptc/kernels/kernel_trans.h>


namespace hptc {

template <typename FloatType,
          TensorOrder ORDER>
class TensorMergedWrapper
    : public TensorWrapper<FloatType, ORDER, MemLayout::COL_MAJOR> {
public:
  TensorMergedWrapper() = delete;

  template <MemLayout ACT_MAJOR>
  TensorMergedWrapper(TensorWrapper<FloatType, ORDER, ACT_MAJOR> &tensor);

  INLINE FloatType &operator[](const TensorIdx * RESTRICT indices);
  INLINE const FloatType &operator[](const TensorIdx * RESTRICT indices) const;
  INLINE FloatType &operator[](TensorIdx **indices);
  INLINE const FloatType &operator[](const TensorIdx **indices) const;

  void merge_idx(const std::unordered_set<TensorOrder> &merge_set);

private:
  TensorOrder merged_order_;
};


template <typename TensorType,
          CoefUsageTrans USAGE>
struct ParamTrans {
  using FloatType = typename TensorType::FLOAT;
  using Deduced = DeducedFloatType<FloatType>;
  constexpr static auto ORDER = TensorType::TENSOR_ORDER;
  constexpr static CoefUsageTrans COEF_USAGE = USAGE;

  ParamTrans(TensorType &input_tensor, TensorType &output_tensor,
      const std::array<TensorOrder, ORDER> &perm, Deduced alpha, Deduced beta);

  INLINE bool is_common_leading();
  INLINE std::pair<TensorOrder, TensorOrder> get_leading();

  TensorMergedWrapper<FloatType, ORDER> input_tensor, output_tensor;
  Deduced alpha, beta;

  TensorOrder perm[ORDER];
  TensorIdx input_stride, output_stride;
  TensorOrder merged_order;
  TensorOrder begin_order_idx;

  // Kernels
  KernelPackTrans<FloatType, USAGE> kn;

private:
  void merge_idx_(const std::array<TensorOrder, ORDER> &perm);
};


/*
 * Import implementation
 */
#include "parameter_trans.tcc"

}

#endif // HPTC_PARAM_PARAMETER_TRANS_H_
