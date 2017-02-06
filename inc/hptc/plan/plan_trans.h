#pragma once
#ifndef HPTC_PLAN_PLAN_TRANS_H_
#define HPTC_PLAN_PLAN_TRANS_H_

#include <vector>
#include <memory>

#include <hptc/types.h>


namespace hptc {

enum PlanTypeTrans {
  PLAN_TRANS_HEUR = 0x0,
  PLAN_TRANS_LOOP = 0x1,
  PLAN_TRANS_PARA = 0x2,
  PLAN_TRANS_PREF = 0x4,
  PLAN_TRANS_AUTO = 0x7
};


template <typename ParamType,
          TensorOrder ORDER,
          PlanTypeTrans TYPE = PLAN_TRANS_HEUR>
class PlanTrans {
public:
  PlanTrans(const std::shared_ptr<ParamTransType> &param);
  CGraphTrans<ParamType, ORDER> *get_graph();

  TensorIdx set_loop_order(const std::vector<TensorOrder> &order);
  TensorIdx set_parallel(const std::vector<TensorOrder> &strategy);
  TensorIdx set_prefetch(const GenNumType distance);

private:
  std::shared_ptr<ParamTransType> param;
};



/*
 * Import implementation
 */
#include "plan_trans.tcc"

}

#endif // HPTC_PLAN_PLAN_TRANS_H_
