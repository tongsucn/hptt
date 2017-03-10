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

template <typename ParamType,
          TensorOrder ORDER>
class OpForTrans {
public:
  OpForTrans();
  OpForTrans(const std::shared_ptr<ParamType> &param,
      const LoopOrderTrans<ORDER> &loop_order,
      const LoopParamTrans<ORDER> &loops);

  OpForTrans(const OpForTrans &loop_data) = delete;
  OpForTrans<ParamType, ORDER> &operator=(const OpForTrans &loop_data) = delete;

  INLINE void init(const std::shared_ptr<ParamType> &param,
      const LoopOrderTrans<ORDER> &loop_order,
      const LoopParamTrans<ORDER> &loops);

  template <typename MacroType>
  INLINE void operator()(MacroType &macro_kernel);

  OpForTrans<ParamType, ORDER> *next;

private:
  void init_disable_();
  void init_loops_(const LoopOrderTrans<ORDER> &loop_order,
      const LoopParamTrans<ORDER> &loops);

  template <typename MacroType,
            GenNumType UNROLL_NUM>
  INLINE void unroller_(GenCounter<UNROLL_NUM>, MacroType &macro_kernel);
  template <typename MacroType>
  INLINE void unroller_(GenCounter<0>, MacroType &macro_kernel);

  std::shared_ptr<ParamType> param_;
  TensorIdx loop_idx_[ORDER];
  TensorIdx *loop_perm_idx_[ORDER];
  TensorIdx loop_begin_[ORDER], loop_end_[ORDER], loop_step_[ORDER];
  TensorIdx loop_order_[ORDER];
};


/*
 * Import implementation
 */
#include "operation_trans.tcc"

}

#endif // HPTC_OPERATIONS_OPERATION_TRANS_H_
