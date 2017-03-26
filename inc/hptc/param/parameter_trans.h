#pragma once
#ifndef HPTC_PARAM_PARAMETER_TRANS_H_
#define HPTC_PARAM_PARAMETER_TRANS_H_

#include <array>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

#include <hptc/compat.h>
#include <hptc/types.h>
#include <hptc/tensor.h>
#include <hptc/util/util_trans.h>
#include <hptc/kernels/kernel_trans.h>


namespace hptc {

template <typename FloatType,
          TensorUInt ORDER>
class TensorMergedWrapper
    : public TensorWrapper<FloatType, ORDER, MemLayout::COL_MAJOR> {
public:
  TensorMergedWrapper() = delete;

  template <MemLayout ACT_MAJOR>
  TensorMergedWrapper(const TensorWrapper<FloatType, ORDER, ACT_MAJOR> &tensor,
      const std::unordered_set<TensorUInt> &merge_set);

  HPTC_INL FloatType &operator[](const TensorIdx * RESTRICT indices);
  HPTC_INL const FloatType &operator[](
      const TensorIdx * RESTRICT indices) const;
  HPTC_INL FloatType &operator[](TensorIdx **indices);
  HPTC_INL const FloatType &operator[](const TensorIdx **indices) const;

private:
  TensorUInt begin_order_idx_, merged_order_;

  TensorUInt merge_idx_(const std::unordered_set<TensorUInt> &merge_set);
};


template <typename TensorType,
          CoefUsageTrans USAGE = CoefUsageTrans::USE_BOTH>
struct ParamTrans {
  // Type alias and constant values
  using FloatType = typename TensorType::Float;
  using Deduced = DeducedFloatType<FloatType>;
  using KernelPack = KernelPackTrans<FloatType, USAGE>;
  using RegTypeFull = typename KernelPack::RegTypeFull;
  using RegTypeHalf = typename KernelPack::RegTypeHalf;
  using RegTypeLinear = typename KernelPack::RegTypeLinear;

  constexpr static auto ORDER = TensorType::TENSOR_ORDER;
  constexpr static CoefUsageTrans COEF_USAGE = USAGE;

private:
  TensorUInt merge_idx_(const std::array<TensorUInt, ORDER> &perm);

  // They need to be initialized before merging
  std::unordered_set<TensorUInt> input_merge_set_, output_merge_set_;

public:
  ParamTrans(const TensorType &input_tensor, TensorType &output_tensor,
      const std::array<TensorUInt, ORDER> &perm, const Deduced alpha,
      const Deduced beta);

  HPTC_INL bool is_common_leading() const;
  HPTC_INL std::pair<TensorUInt, TensorUInt> get_leading() const;
  HPTC_INL void set_coef(const Deduced alpha, const Deduced beta);

  std::array<TensorUInt, ORDER> perm;
  Deduced alpha, beta;
  RegTypeFull reg_alpha_full, reg_beta_full;
  RegTypeHalf reg_alpha_half, reg_beta_half;
  RegTypeLinear reg_alpha_linear, reg_beta_linear;

  TensorIdx input_stride, output_stride;
  TensorUInt begin_order_idx, merged_order;

  // Put the merged tensors here, they must be initialized after merging
  const TensorMergedWrapper<FloatType, ORDER> input_tensor;
  TensorMergedWrapper<FloatType, ORDER> output_tensor;

  // Kernels
  const KernelPack &kn;
};


/*
 * Import implementation
 */
#include "parameter_trans.tcc"

}

#endif // HPTC_PARAM_PARAMETER_TRANS_H_
