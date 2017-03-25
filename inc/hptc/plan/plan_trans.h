#pragma once
#ifndef HPTC_PLAN_PLAN_TRANS_H_
#define HPTC_PLAN_PLAN_TRANS_H_

#include <cfloat>

#include <vector>
#include <memory>
#include <utility>
#include <initializer_list>

#include <hptc/types.h>
#include <hptc/util/util.h>
#include <hptc/util/util_trans.h>
#include <hptc/cgraph/cgraph_trans.h>
#include <hptc/plan/plan_trans_util.h>


namespace hptc {

/*
 * Definition of class PlanTrans
 */
template <typename ParamType>
class PlanTrans {
public:
  PlanTrans(const std::shared_ptr<ParamType> &param,
      const TensorUInt num_threads,
      const TensorInt tune_loop_num, const TensorInt tune_para_num,
      const TensorInt heur_loop_num, const TensorInt heur_para_num,
      const double tuning_timeout_ms, const TensorUInt tune_times = 5);

  PlanTrans(const PlanTrans &plan) = delete;
  PlanTrans<ParamType> &operator=(const PlanTrans &plan) = delete;

  ~PlanTrans() = default;

  CGraphTrans<ParamType> *get_graph();

private:
  using Descriptor = typename CGraphTrans<ParamType>::Descriptor;

  Descriptor tuning_(const std::vector<Descriptor> &descriptors,
      const double tuning_timeout, const TensorUInt tune_times);

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
