#pragma once
#ifndef HPTC_PLAN_PLAN_TRANS_H_
#define HPTC_PLAN_PLAN_TRANS_H_

#include <vector>
#include <memory>
#include <utility>
#include <cmath>
#include <initializer_list>

#include <hptc/types.h>
#include <hptc/util/util.h>
#include <hptc/cgraph/cgraph_trans.h>
#include <hptc/plan/plan_trans_util.h>
#include <hptc/param/parameter_trans.h>


namespace hptc {

template <typename ParamType>
class PlanTrans {
public:
  PlanTrans(const std::shared_ptr<ParamType> &param, TensorIdx tune_loop_num,
      TensorIdx tune_para_num, TensorIdx heur_loop_num, TensorIdx heur_para_num,
      GenNumType thread_num = 0, GenNumType tune_times = 5);

  PlanTrans(const PlanTrans &plan) = delete;
  PlanTrans<ParamType> &operator=(const PlanTrans &plan) = delete;

  ~PlanTrans() = default;

  CGraphTrans<ParamType> *get_graph();

private:
  using Descriptor = typename CGraphTrans<ParamType>::Descriptor;

  Descriptor tuning_(const std::vector<Descriptor> &descriptors,
      GenNumType tune_times);

  std::shared_ptr<ParamType> param_;
  PlanTransOptimizer<ParamType> optimizer_;
  Descriptor optimal_descriptor_;
};


/*
 * Import implementation
 */
#include "plan_trans.tcc"

}

#endif // HPTC_PLAN_PLAN_TRANS_H_
