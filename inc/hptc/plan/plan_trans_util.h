#pragma once
#ifndef HPTC_PLAN_PLAN_TRANS_UTIL_H_
#define HPTC_PLAN_PLAN_TRANS_UTIL_H_

#include <cmath>
#include <vector>
#include <stack>
#include <queue>
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
class PlanTransOptimizer {
public:
  using Descriptor = typename CGraphTrans<ParamType>::Descriptor;
  static constexpr auto ORDER = ParamType::ORDER;

  PlanTransOptimizer(const std::shared_ptr<ParamType> &param,
      TensorIdx tune_loop_num, TensorIdx tune_para_num, TensorIdx heur_loop_num,
      TensorIdx heur_para_num, GenNumType thread_num);

  std::vector<Descriptor> get_optimal() const;

private:
  struct LoopParaStrategy_ {
    LoopParaStrategy_(TensorOrder size, GenNumType th_num, TensorOrder loop_idx)
        : size(size), th_num(th_num), loop_idx(loop_idx) {}
    TensorOrder size;
    GenNumType th_num;
    TensorOrder loop_idx;
  };

  void init_(TensorIdx tune_loop_num, TensorIdx tune_para_num,
      TensorIdx heur_loop_num, TensorIdx heur_para_num);
  void init_config_();
  void init_loop_evaluator_param_();
  void init_parallel_evaluator_param_();

  void init_loop_rule_();
  void init_loop_heur_(const TensorIdx tune_num, const TensorIdx heur_num);

  void init_threads_();

  void init_vec_general_();
  void init_vec_deploy_kernels_(const KernelTypeTrans kn_type,
      const GenNumType kn_cont_size, const GenNumType kn_ncont_size,
      const TensorOrder cont_begin_pos, const TensorOrder ncont_begin_pos,
      const TensorOrder cont_offset_size, const TensorOrder ncont_offset_size,
      const bool is_linh = false);
  void init_vec_common_leading_();

  void init_parallel_rule_general_();
  void init_parallel_rule_common_leading_();
  void init_parallel_heur_(const TensorIdx tune_num, const TensorIdx heur_num);

  double heur_loop_evaluator_(
      const LoopOrderTrans<ORDER> &target_loop_order) const;
  double heur_parallel_evaluator_(
      const ParaStrategyTrans<ORDER> &target_para) const;

  std::vector<Descriptor> gen_candidates_() const;


  std::shared_ptr<ParamType> param_;
  GenNumType threads_;
  const TensorOrder in_ld_idx_, out_ld_idx_;
  std::unordered_map<GenNumType, GenNumType> th_factor_map_;
  ParaStrategyTrans<ORDER> avail_parallel_;

  std::vector<LoopOrderTrans<ORDER>> loop_order_candidates_;
  std::vector<ParaStrategyTrans<ORDER>> parallel_strategy_candidates_;
  Descriptor template_descriptor_;

  // Parameters for loop order heuristics
  double heur_loop_penalty_begin, heur_loop_penalty_step;
  double heur_loop_importance_begin, heur_loop_importance_scale;
  double heur_loop_input_penalty_factor, heur_loop_output_penalty_factor;
  double heur_loop_in_ld_award, heur_loop_out_ld_award;

  // Parameters for parallelization heuristics
  double heur_para_penalty_factor_cl, heur_para_penalty_factor_inld,
         heur_para_penalty_factor_outld, heur_para_cost_begin;
  GenNumType heur_para_max_penalty_threads;
};


/*
 * Import implementation
 */
#include "plan_trans_util.tcc"

}

#endif // HPTC_PLAN_PLAN_TRANS_UTIL_H_
