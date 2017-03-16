#pragma once
#ifndef HPTC_PLAN_PLAN_TRANS_UTIL_H_
#define HPTC_PLAN_PLAN_TRANS_UTIL_H_

#include <cmath>
#include <vector>
#include <stack>
#include <queue>
#include <deque>
#include <memory>
#include <utility>
#include <numeric>
#include <algorithm>
#include <functional>
#include <unordered_set>

#include <omp.h>

#include <hptc/types.h>
#include <hptc/util.h>
#include <hptc/config/config_trans.h>
#include <hptc/cgraph/cgraph_trans.h>
#include <hptc/param/parameter_trans.h>


namespace hptc {

template <typename ParamType>
using Graph = CGraphTrans<ParamType>;

template <typename ParamType>
using Descriptor = typename Graph<ParamType>::CGraphTransDescriptor;


template <typename ParamType>
class PlanTransOptimizer {
public:
  static constexpr auto ORDER = ParamType::ORDER;

  PlanTransOptimizer(const std::shared_ptr<ParamType> &param,
      GenNumType thread_num = 0);

  std::vector<Descriptor<ParamType>> get_optimal(TensorIdx heur_loop_num,
      TensorIdx heur_para_num, TensorIdx tune_loop_num,
      TensorIdx tune_para_num) const;

private:
  struct LoopParaStrategy_ {
    LoopParaStrategy_(TensorOrder size, GenNumType th_num, TensorOrder loop_idx)
        : size(size), th_num(th_num), loop_idx(loop_idx) {}
    TensorOrder size;
    GenNumType th_num;
    TensorOrder loop_idx;
  };

  void init_();
  void init_config_();
  void init_loop_evaluator_param_();
  void init_parallel_evaluator_param_();

  void init_threads_();

  void init_vec_();
  void init_vec_deploy_kernels_(const KernelTypeTrans kn_type,
      const GenNumType kn_cont_size, const GenNumType kn_ncont_size,
      const TensorOrder cont_begin_pos, const TensorOrder ncont_begin_pos,
      const TensorOrder cont_offset_size, const TensorOrder ncont_offset_size,
      const bool is_linh = false);
  void init_vec_common_leading_();

  void init_loop_();
  void init_parallel_();
  void init_parallel_common_leading_();

  std::vector<LoopOrderTrans<ParamType::ORDER>> heur_loop_explorer_(
      const TensorIdx heur_num, TensorIdx tune_num) const;
  double heur_loop_evaluator_(
      const LoopOrderTrans<ORDER> &target_loop_order) const;

  std::vector<ParaStrategyTrans<ParamType::ORDER>> heur_parallel_explorer_(
      const TensorIdx heur_num, TensorIdx tune_num) const;
  double heur_parallel_evaluator_(
      const ParaStrategyTrans<ORDER> &target_para) const;

  std::vector<Descriptor<ParamType>> gen_candidates_(
      const std::vector<LoopOrderTrans<ORDER>> &loop_orders,
      const std::vector<ParaStrategyTrans<ORDER>> &parallel_strategies) const;
  void parallelize_(Descriptor<ParamType> &descriptor) const;


  std::shared_ptr<ParamType> param_;
  GenNumType threads_;

  Descriptor<ParamType> descriptor_;
  std::unordered_map<GenNumType, GenNumType> th_fact_map_;
  ParaStrategyTrans<ORDER> avail_parallel_;
  ParaStrategyTrans<ORDER> parallel_template_;

  // Parameters for loop order heuristics
  double penalty_begin, penalty_step;
  double importance_begin, importance_scale;
  double input_penalty_factor, output_penalty_factor;
  double in_ld_award, out_ld_award;

  // Parameters for parallelization heuristics
  double penalty_factor_cl, penalty_factor_inld, penalty_factor_outld;
  GenNumType max_penalty_threads;
};


/*
 * Import implementation
 */
#include "plan_trans_util.tcc"

}

#endif // HPTC_PLAN_PLAN_TRANS_UTIL_H_
