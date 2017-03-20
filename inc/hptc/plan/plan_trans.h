#pragma once
#ifndef HPTC_PLAN_PLAN_TRANS_H_
#define HPTC_PLAN_PLAN_TRANS_H_

#include <vector>
#include <memory>
#include <utility>
#include <cmath>
#include <initializer_list>

#include <hptc/types.h>
#include <hptc/util.h>
#include <hptc/cgraph/cgraph_trans.h>
#include <hptc/plan/plan_trans_util.h>
#include <hptc/param/parameter_trans.h>


namespace hptc {

template <typename ParamType>
class PlanTrans {
public:
  static constexpr auto ORDER = ParamType::ORDER;

  PlanTrans(const std::shared_ptr<ParamType> &param, GenNumType thread_num = 0);

  PlanTrans(const PlanTrans &plan) = delete;
  PlanTrans<ParamType> &operator=(const PlanTrans &plan) = delete;

  ~PlanTrans() = default;

  Graph<ParamType> *get_graph(TensorIdx heur_num = 0, TensorIdx tune_num = 0,
      GenNumType tune_times = 10);
  Graph<ParamType> *get_graph(std::initializer_list<TensorIdx> loop_param,
      std::initializer_list<TensorIdx> parallel_param,
      GenNumType tune_times = 10);

private:
  Graph<ParamType> *tuning_(
      const std::vector<Descriptor<ParamType>> &descriptors,
      GenNumType tune_times);

  std::shared_ptr<ParamType> param_;
  PlanTransOptimizer<ParamType> optimizer_;
};


/*
 * Import implementation
 */
#include "plan_trans.tcc"

}

#endif // HPTC_PLAN_PLAN_TRANS_H_
