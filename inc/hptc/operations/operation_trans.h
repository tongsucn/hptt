#pragma once
#ifndef HPTC_OPERATIONS_OPERATION_TRANS_H_
#define HPTC_OPERATIONS_OPERATION_TRANS_H_

#include <array>
#include <vector>
#include <memory>
#include <algorithm>

#include <hptc/types.h>
#include <hptc/util.h>
#include <hptc/param/parameter_trans.h>


namespace hptc {

template <typename ParamType,
          TensorOrder ORDER>
class OpForTrans {
public:
  OpForTrans();
  OpForTrans(const std::shared_ptr<ParamType> &param);

  OpForTrans(const OpForTrans &loop_data);
  OpForTrans<ParamType, ORDER> &operator=(const OpForTrans &loop_data);

  INLINE void init(const std::shared_ptr<ParamType> &param);

  template <typename MacroType>
  INLINE void operator()(MacroType &macro_kernel);

  INLINE void set_begin(TensorIdx begin_val, TensorIdx idx);
  INLINE void set_end(TensorIdx end_val, TensorIdx idx);
  INLINE void set_step(TensorIdx step_val, TensorIdx idx);

  INLINE void set_order(const std::vector<TensorOrder> &order);
  INLINE void set_pass(TensorOrder order);

  INLINE const TensorIdx *get_order() const;

  OpForTrans<ParamType, ORDER> *next;

private:
  void init_disable_();
  void init_perm_idx_();

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
