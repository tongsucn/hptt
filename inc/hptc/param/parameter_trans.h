#pragma once
#ifndef HPTC_PARAM_PARAMETER_TRANS_H_
#define HPTC_PARAM_PARAMETER_TRANS_H_

#include <array>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

#include <hptc/types.h>
#include <hptc/tensor.h>
#include <hptc/config/config_trans.h>
#include <hptc/kernels/macro_kernel_trans.h>


namespace hptc {

template <typename FloatType,
          TensorOrder ORDER,
          MemLayout LAYOUT = MemLayout::COL_MAJOR>
class TensorMergedWrapper : public TensorWrapper<FloatType, ORDER, LAYOUT> {
public:
  TensorMergedWrapper() = delete;
  TensorMergedWrapper(const TensorWrapper<FloatType, ORDER, LAYOUT> &wrapper);

  INLINE FloatType &operator[](const TensorIdx * RESTRICT indices);
  INLINE const FloatType &operator[](const TensorIdx * RESTRICT indices) const;
  INLINE FloatType &operator[](TensorIdx **indices);
  INLINE const FloatType &operator[](const TensorIdx **indices) const;

  INLINE TensorOrder get_leading();
  void merge_idx(const std::unordered_set<TensorOrder> &merge_set);

private:
  TensorOrder merged_order_;
};


template <typename FloatType,
          TensorOrder ORDER,
          CoefUsageTrans USAGE,
          MemLayout LAYOUT = MemLayout::COL_MAJOR>
struct ParamTrans {
  ParamTrans(const TensorWrapper<FloatType, ORDER, LAYOUT> &input_tensor,
      const TensorWrapper<FloatType, ORDER, LAYOUT> &output_tensor,
      const std::array<TensorOrder, ORDER> &perm,
      DeducedFloatType<FloatType> alpha, DeducedFloatType<FloatType> beta);

  INLINE TensorIdx perm_type();

  using DataType = FloatType;
  constexpr static CoefUsageTrans COEF_USAGE = USAGE;
  constexpr static MemLayout MEM_LAYOUT = LAYOUT;

  TensorWrapper<FloatType, ORDER, LAYOUT> org_input_tensor, org_output_tensor;
  TensorMergedWrapper<FloatType, ORDER, LAYOUT> input_tensor, output_tensor;
  DeducedFloatType<FloatType> alpha, beta;

  TensorOrder perm[ORDER];
  TensorIdx input_stride, output_stride;
  TensorOrder merged_order;

  // Kernels
  MacroTransVecFullBig<FloatType, USAGE>        kn_fb;
  MacroTransVecFullVertical<FloatType, USAGE>   kn_fv;
  MacroTransVecFullHorizontal<FloatType, USAGE> kn_fh;
  MacroTransVecFullSmall<FloatType, USAGE>      kn_fs;
  MacroTransVecHalfVertical<FloatType, USAGE>   kn_hv;
  MacroTransVecHalfHorizontal<FloatType, USAGE> kn_hh;
  MacroTransVecHalfSmall<FloatType, USAGE>      kn_hs;
  MacroTransScalar<FloatType, USAGE>            kn_sc;

private:
  void merge_idx_(const std::array<TensorOrder, ORDER> &perm);
};


/*
 * Import implementation
 */
#include "parameter_trans.tcc"

}

#endif // HPTC_PARAM_PARAMETER_TRANS_H_
