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
      TensorIdx tune_para_num) const;

private:
  using Loop_ = std::array<std::pair<TensorIdx, GenNumType>, ORDER>;
  using Factor_ = std::vector<std::pair<GenNumType, GenNumType>>;

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
  template <typename Cmp>
  void init_parallel_loop_(Loop_ &loops, Factor_ &fact_map,
      const TensorOrder input_ld_idx, const TensorOrder output_ld_idx,
      Cmp cmp) const;

  std::vector<LoopOrder<ORDER>> heur_loop_explorer_(const TensorIdx heur_num,
      const TensorIdx tune_num) const;
  double heur_loop_evaluator_(const LoopOrder<ORDER> &target_loop_order) const;

  std::vector<std::vector<GenNumType>> heur_parallel_explorer_(
      const TensorIdx heur_num, const TensorIdx tune_num) const;
  double heur_parallel_evaluator_(
      const std::vector<GenNumType> &target_para) const;

  std::vector<CGraphTransDescriptor<ORDER>> gen_candidates_(
      const std::vector<LoopOrder<ORDER>> &loop_orders,
      const std::vector<std::vector<GenNumType>> &parallel_strategies) const;
  void parallelize_(CGraphTransDescriptor<ORDER> &descriptor) const;


  std::shared_ptr<ParamType> param_;
  GenNumType threads_;

  CGraphTransDescriptor<ORDER> descriptor_;
  Factor_ th_fact_map_;

  // Parameters for loop order heuristics
  double penalty_begin, penalty_step;
  double importance_begin, importance_scale;
  double input_penalty_factor, output_penalty_factor;

  // Parameters for parallelization heuristics
};


/*
 * Import implementation
 */
#include "plan_trans_util.tcc"

}

#endif // HPTC_PLAN_PLAN_TRANS_UTIL_H_
