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
  using KernelPack = KernelPackTrans<FloatType, USAGE>;
  using RegTypeFull = typename KernelPack::RegTypeFull;
  using RegTypeHalf = typename KernelPack::RegTypeHalf;
  using RegTypeLinear = typename KernelPack::RegTypeLinear;

  constexpr static auto ORDER = TensorType::TENSOR_ORDER;
  constexpr static CoefUsageTrans COEF_USAGE = USAGE;

  ParamTrans(TensorType &input_tensor, TensorType &output_tensor,
      const std::array<TensorOrder, ORDER> &perm, const Deduced alpha,
      const Deduced beta);

  INLINE bool is_common_leading();
  INLINE std::pair<TensorOrder, TensorOrder> get_leading();
  INLINE void set_coef(const Deduced alpha, const Deduced beta);


  TensorMergedWrapper<FloatType, ORDER> input_tensor, output_tensor;
  std::array<TensorOrder, ORDER> perm;
  Deduced alpha, beta;
  RegTypeFull reg_alpha_full, reg_beta_full;
  RegTypeHalf reg_alpha_half, reg_beta_half;
  RegTypeLinear reg_alpha_linear, reg_beta_linear;

  TensorIdx input_stride, output_stride;
  TensorOrder merged_order;
  TensorOrder begin_order_idx;

  // Kernels
  const KernelPack &kn;

private:
  TensorOrder merge_idx_(const std::array<TensorOrder, ORDER> &perm);
};


/*
 * Import implementation
 */
#include "parameter_trans.tcc"

}

#endif // HPTC_PARAM_PARAMETER_TRANS_H_
