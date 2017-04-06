#pragma once
#ifndef HPTT_OPERATIONS_OPERATION_TRANS_H_
#define HPTT_OPERATIONS_OPERATION_TRANS_H_

#include <array>
#include <algorithm>

#include <hptt/types.h>
#include <hptt/arch/compat.h>
#include <hptt/util/util.h>
#include <hptt/util/util_trans.h>


namespace hptt {

template <TensorUInt ORDER>
class OpForTrans {
public:
  OpForTrans();
  OpForTrans(const LoopOrderTrans<ORDER> &loop_order,
      const LoopParamTrans<ORDER> &loops, const TensorUInt begin_order_idx,
      const std::array<TensorUInt, ORDER> &perm);

  OpForTrans(const OpForTrans &loop_data) = delete;
  OpForTrans<ORDER> &operator=(const OpForTrans &loop_data) = delete;

  HPTT_INL void init(const LoopOrderTrans<ORDER> &loop_order,
      const LoopParamTrans<ORDER> &loops, const TensorUInt begin_order_idx,
      const std::array<TensorUInt, ORDER> &perm);

  template <typename MacroType,
            typename TensorType>
  HPTT_INL void exec(const MacroType &macro_kernel,
      const TensorType &input_tensor, TensorType &output_tensor,
      const TensorIdx stride_in_outld, const TensorIdx stride_out_inld);

  OpForTrans<ORDER> *next;

private:
  void init_disable_();
  void init_loops_(const LoopOrderTrans<ORDER> &loop_order,
      const LoopParamTrans<ORDER> &loops);

  template <typename MacroType,
            typename TensorType,
            TensorUInt UNROLL_NUM>
  void unroller_(GenCounter<UNROLL_NUM>, const MacroType &macro_kernel,
      const TensorType &input_tensor, TensorType &output_tensor,
      const TensorIdx stride_in_outld, const TensorIdx stride_out_inld);
  template <typename MacroType,
            typename TensorType>
  void unroller_(GenCounter<0>, const MacroType &macro_kernel,
      const TensorType &input_tensor, TensorType &output_tensor,
      const TensorIdx stride_in_outld, const TensorIdx stride_out_inld);

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

#endif // HPTT_OPERATIONS_OPERATION_TRANS_H_
