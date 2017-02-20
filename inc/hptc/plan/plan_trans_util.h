#pragma once
#ifndef HPTC_PLAN_PLAN_TRANS_UTIL_H_
#define HPTC_PLAN_PLAN_TRANS_UTIL_H_

#include <vector>
#include <memory>
#include <utility>
#include <numeric>
#include <algorithm>

#include <omp.h>

#include <hptc/types.h>
#include <hptc/util.h>
#include <hptc/config/config_trans.h>
#include <hptc/cgraph/cgraph_trans.h>
#include <hptc/param/parameter_trans.h>


namespace hptc {

template <typename ParamType,
          TensorOrder ORDER>
class PlanTransOptimizer {
public:
  PlanTransOptimizer(const std::shared_ptr<ParamType> &param,
      GenNumType thread_num = 0);

  CGraphTransDescriptor<ORDER> get_optimal(TensorIdx heur_loop_num = 0,
      TensorIdx heur_para_num = 0);

private:
  struct Loop_ {
    TensorIdx size;
    GenNumType thread_num;
    TensorOrder org_idx;
  };

  void init_();
  void init_thread_num_();
  void init_vec_();
  bool init_vec_kernels_(LoopParam<ORDER> &loop, GenNumType cont_len,
      GenNumType ncont_len, TensorOrder &cont_rest, TensorOrder &ncont_rest,
      TensorIdx &cont_begin, TensorIdx &ncont_begin);
  void init_loop_();
  void init_loop_evaluator_param_();
  void init_parallel_();

  LoopOrder<ORDER> heur_loop_explorer_(TensorIdx num = 0);
  double heur_loop_evaluator_(const LoopOrder<ORDER> &target_loop_order);
  void heur_parallel_explorer_(TensorIdx num = 0);

  std::shared_ptr<ParamType> param_;
  GenNumType threads_;
  std::vector<GenNumType> strategy_;
  CGraphTransDescriptor<ORDER> descriptor_;

  // Parameters for loop order cost calculation
  double penalty_begin, penalty_step;
  double importance_begin, importance_scale;
  double input_penalty_factor, output_penalty_factor;
};


template <typename ParamType,
          TensorOrder ORDER>
class PlanTransParallelizer {
public:
  PlanTransParallelizer(const std::shared_ptr<ParamType> &param);

  LoopOrder<ORDER> operator()(CGraphTransDescriptor<ORDER> &descriptor,
      GenNumType threads = 0);

private:
  LoopOrder<ORDER> calc_depth_(const CGraphTransDescriptor<ORDER> &des,
      std::vector<GenNumType> &strategy);
  void parallelize_(CGraphTransDescriptor<ORDER> &des,
      const LoopOrder<ORDER> &loop_order,
      const std::vector<GenNumType> &strategy);

  std::shared_ptr<ParamType> param_;
};


/*
 * Import implementation
 */
#include "plan_trans_util.tcc"

}

#endif // HPTC_PLAN_PLAN_TRANS_UTIL_H_
