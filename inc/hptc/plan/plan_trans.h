#pragma once
#ifndef HPTC_PLAN_PLAN_TRANS_H_
#define HPTC_PLAN_PLAN_TRANS_H_

#include <array>
#include <vector>
#include <memory>
#include <utility>
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
  PlanTrans(const std::shared_ptr<ParamType> &param);

  PlanTrans(const PlanTrans &plan) = delete;
  PlanTrans<ParamType, ORDER> &operator=(const PlanTrans &plan) = delete;

  ~PlanTrans() = default;

  CGraphTrans<ParamType, ORDER> *get_graph(
      PlanTypeTrans plan_type = PLAN_TRANS_AUTO);

private:
  CGraphTrans<ParamType, ORDER> *cgraph_auto_();
  CGraphTrans<ParamType, ORDER> *cgraph_heur_();

  std::shared_ptr<ParamType> param_;
  PlanTransVectorizer<ParamType, ORDER> vectorizer_;
  PlanTransParallelizer<ParamType, ORDER> parallelizer_;
};



/*
 * Import implementation
 */
#include "plan_trans.tcc"

}

#endif // HPTC_PLAN_PLAN_TRANS_H_
