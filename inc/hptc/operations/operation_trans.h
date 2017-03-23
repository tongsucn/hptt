#pragma once
#ifndef HPTC_OPERATIONS_OPERATION_TRANS_H_
#define HPTC_OPERATIONS_OPERATION_TRANS_H_

#include <array>
#include <algorithm>

#include <hptc/types.h>
#include <hptc/util/util.h>
#include <hptc/util/util_trans.h>


namespace hptc {

template <TensorUInt ORDER>
class OpForTrans {
public:
  OpForTrans();
  OpForTrans(const LoopOrderTrans<ORDER> &loop_order,
      const LoopParamTrans<ORDER> &loops, const TensorUInt begin_order_idx,
      const std::array<TensorUInt, ORDER> &perm);

  OpForTrans(const OpForTrans &loop_data) = delete;
  OpForTrans<ORDER> &operator=(const OpForTrans &loop_data) = delete;

  INLINE void init(const LoopOrderTrans<ORDER> &loop_order,
      const LoopParamTrans<ORDER> &loops, const TensorUInt begin_order_idx,
      const std::array<TensorUInt, ORDER> &perm);

  template <typename MacroType,
            typename TensorType,
            typename RegType>
  INLINE void operator()(MacroType &macro_kernel,
      const TensorType &input_tensor, TensorType &output_tensor,
      const TensorIdx input_stride, const TensorIdx output_stride,
      const RegType &reg_alpha, const RegType &reg_beta);

  OpForTrans<ORDER> *next;

private:
  void init_disable_();
  void init_loops_(const LoopOrderTrans<ORDER> &loop_order,
      const LoopParamTrans<ORDER> &loops);

  template <typename MacroType,
            typename TensorType,
            typename RegType,
            TensorUInt UNROLL_NUM>
  INLINE void unroller_(GenCounter<UNROLL_NUM>, MacroType &macro_kernel,
      const TensorType &input_tensor, TensorType &output_tensor,
      const TensorIdx input_stride, const TensorIdx output_stride,
      const RegType &reg_alpha, const RegType &reg_beta);
  template <typename MacroType,
            typename TensorType,
            typename RegType>
  INLINE void unroller_(GenCounter<0>, MacroType &macro_kernel,
      const TensorType &input_tensor, TensorType &output_tensor,
      const TensorIdx input_stride, const TensorIdx output_stride,
      const RegType &reg_alpha, const RegType &reg_beta);

  TensorIdx loop_idx_[ORDER];
  TensorIdx *loop_perm_idx_[ORDER];
  TensorIdx loop_begin_[ORDER], loop_end_[ORDER], loop_step_[ORDER];
  TensorIdx loop_order_[ORDER];
};


/*
 * Import implementation and explicit template instantiation declaration
 */
#include "operation_trans.tcc"

}

#endif // HPTC_OPERATIONS_OPERATION_TRANS_H_
