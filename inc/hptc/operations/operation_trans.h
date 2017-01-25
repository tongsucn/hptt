#pragma once
#ifndef HPTC_OPERATIONS_OPERATION_TRANS_H_
#define HPTC_OPERATIONS_OPERATION_TRANS_H_

#include <memory>
#include <algorithm>

#include <hptc/types.h>
#include <hptc/util.h>
#include <hptc/param/parameter_trans.h>


namespace hptc {

template <TensorOrder ORDER,
          typename ParamType>
class OpForTransData {
public:
  OpForTransData(std::shared_ptr<ParamType> &param);

  INLINE void set_begin(TensorIdx begin_val, TensorIdx idx);
  INLINE void set_end(TensorIdx end_val, TensorIdx idx);
  INLINE void set_step(TensorIdx step_val, TensorIdx idx);

protected:
  std::shared_ptr<ParamType> param_;
  TensorIdx loop_idx_[ORDER];
  TensorIdx *loop_perm_idx_[ORDER];
  TensorIdx loop_begin_[ORDER], loop_end_[ORDER], loop_step_[ORDER];
};


template <TensorOrder ORDER,
          typename ParamType>
class OpForTrans final : public OpForTransData<ORDER, ParamType> {
public:
  OpForTrans(std::shared_ptr<ParamType> &param);

  template <typename MacroType>
  INLINE void operator()(MacroType &macro_kernel);

  OpForTrans<ORDER, ParamType> *next;

private:
  template <typename MacroType,
            GenNumType UNROLL_NUM>
  INLINE void unroller_(GenCounter<UNROLL_NUM>, MacroType &macro_kernel);
  template <typename MacroType>
  INLINE void unroller_(GenCounter<0>, MacroType &macro_kernel);
};


/*
 * Import implementation
 */
#include "operation_trans.tcc"

}

#endif // HPTC_OPERATIONS_OPERATION_TRANS_H_
