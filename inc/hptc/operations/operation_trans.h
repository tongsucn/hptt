#pragma once
#ifndef HPTC_OPERATIONS_OPERATION_TRANS_H_
#define HPTC_OPERATIONS_OPERATION_TRANS_H_

#include <array>
#include <memory>
#include <algorithm>

#include <hptc/types.h>
#include <hptc/util.h>
#include <hptc/config/config_trans.h>
#include <hptc/param/parameter_trans.h>


namespace hptc {

template <TensorOrder ORDER>
class OpForTrans {
public:
  OpForTrans();
  OpForTrans(const LoopOrderTrans<ORDER> &loop_order,
      const LoopParamTrans<ORDER> &loops, const TensorOrder begin_order_idx,
      const TensorOrder *perm);

  OpForTrans(const OpForTrans &loop_data) = delete;
  OpForTrans<ORDER> &operator=(const OpForTrans &loop_data) = delete;

  INLINE void init(const LoopOrderTrans<ORDER> &loop_order,
      const LoopParamTrans<ORDER> &loops, const TensorOrder begin_order_idx,
      const TensorOrder *perm);

  template <typename MacroType,
            typename TensorType>
  INLINE void operator()(MacroType &macro_kernel,
      const TensorType &input_tensor, TensorType &output_tensor,
      const TensorIdx input_stride, const TensorIdx output_stride);

  OpForTrans<ORDER> *next;

private:
  void init_disable_();
  void init_loops_(const LoopOrderTrans<ORDER> &loop_order,
      const LoopParamTrans<ORDER> &loops);

  template <typename MacroType,
            typename TensorType,
            GenNumType UNROLL_NUM>
  INLINE void unroller_(GenCounter<UNROLL_NUM>, MacroType &macro_kernel,
      const TensorType &input_tensor, TensorType &output_tensor,
      const TensorIdx input_stride, const TensorIdx output_stride);
  template <typename MacroType,
            typename TensorType>
  INLINE void unroller_(GenCounter<0>, MacroType &macro_kernel,
      const TensorType &input_tensor, TensorType &output_tensor,
      const TensorIdx input_stride, const TensorIdx output_stride);

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
