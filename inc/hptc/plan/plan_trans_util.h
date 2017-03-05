#pragma once
#ifndef HPTC_PLAN_PLAN_TRANS_UTIL_H_
#define HPTC_PLAN_PLAN_TRANS_UTIL_H_

#include <vector>
#include <queue>
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

  std::vector<CGraphTransDescriptor<ORDER>> get_optimal(TensorIdx heur_loop_num,
      TensorIdx heur_para_num, TensorIdx tune_loop_num,
      TensorIdx tune_para_num);

private:
  void init_();
  void init_thread_num_();
  void init_vec_();
  void init_vec_kernels_(LoopParam<ORDER> &loop, const GenNumType kn_cont_len,
      const GenNumType kn_ncont_len, TensorOrder &cont_rest,
      TensorOrder &ncont_rest, bool is_sv = false);
  void init_vec_cl_();
  void init_vec_kernels_cl_(LoopParam<ORDER> &loop, const GenNumType kn_len,
      const TensorOrder input_leading, TensorOrder &cont_rest);

  void init_loop_();
  void init_loop_evaluator_param_();
  void init_parallel_();

  std::vector<LoopOrder<ORDER>> heur_loop_explorer_(const TensorIdx heur_num,
      const TensorIdx tune_num);
  double heur_loop_evaluator_(const LoopOrder<ORDER> &target_loop_order);
  void heur_parallel_explorer_(const TensorIdx heur_num,
      const TensorIdx tune_num);

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
