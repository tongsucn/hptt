#pragma once
#ifndef HPTC_PLAN_PLAN_TRANS_H_
#define HPTC_PLAN_PLAN_TRANS_H_

#include <array>
#include <vector>
#include <memory>
#include <utility>
#include <numeric>
#include <algorithm>

#include <omp.h>

#include <iostream>

#include <hptc/types.h>
#include <hptc/config/config_trans.h>
#include <hptc/cgraph/cgraph_trans.h>
#include <hptc/param/parameter_trans.h>


namespace hptc {

enum PlanTypeTrans {
  PLAN_TRANS_AUTO = 0x0,
  PLAN_TRANS_LOOP = 0x1,
  PLAN_TRANS_PARA = 0x2,
  PLAN_TRANS_HEUR = 0x3
};


template <typename ParamType,
          TensorOrder ORDER>
class PlanTrans {
public:
  PlanTrans(const std::shared_ptr<ParamType> &param);

  PlanTrans(const PlanTrans &plan) = delete;
  PlanTrans<ParamType, ORDER> &operator=(const PlanTrans &plan) = delete;

  CGraphTrans<ParamType, ORDER> *get_graph(
      PlanTypeTrans plan_type = PLAN_TRANS_AUTO);

private:
  struct LoopNode {
    TensorIdx size;
    GenNumType thread_num;
    TensorOrder org_idx;
  };

  CGraphTrans<ParamType, ORDER> *cgraph_auto_();
  CGraphTrans<ParamType, ORDER> *cgraph_heur_();

  std::shared_ptr<ParamType> param_;
};



/*
 * Import implementation
 */
#include "plan_trans.tcc"

}

#endif // HPTC_PLAN_PLAN_TRANS_H_
