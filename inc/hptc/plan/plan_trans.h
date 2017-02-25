#pragma once
#ifndef HPTC_PLAN_PLAN_TRANS_H_
#define HPTC_PLAN_PLAN_TRANS_H_

#include <array>
#include <vector>
#include <memory>
#include <utility>
#include <cmath>
#include <algorithm>

#include <hptc/types.h>
#include <hptc/util.h>
#include <hptc/cgraph/cgraph_trans.h>
#include <hptc/plan/plan_trans_util.h>
#include <hptc/param/parameter_trans.h>


namespace hptc {

template <typename ParamType,
          TensorOrder ORDER>
class PlanTrans {
public:
  PlanTrans(const std::shared_ptr<ParamType> &param, GenNumType thread_num = 0);

  PlanTrans(const PlanTrans &plan) = delete;
  PlanTrans<ParamType, ORDER> &operator=(const PlanTrans &plan) = delete;

  ~PlanTrans() = default;

  CGraphTrans<ParamType, ORDER> *get_graph(TensorIdx heur_num = 0,
      TensorIdx tune_num = 0, GenNumType tune_timing = 10);

private:
  using Graph = CGraphTrans<ParamType, ORDER>;

  Graph *tuning_(const std::vector<CGraphTransDescriptor<ORDER>> &descriptors,
      GenNumType tune_timing);

  std::shared_ptr<ParamType> param_;
  PlanTransOptimizer<ParamType, ORDER> optimizer_;
};


/*
 * Import implementation
 */
#include "plan_trans.tcc"

}

#endif // HPTC_PLAN_PLAN_TRANS_H_
