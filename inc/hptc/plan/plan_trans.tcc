#pragma once
#ifndef HPTC_PLAN_PLAN_TRANS_TCC_
#define HPTC_PLAN_PLAN_TRANS_TCC_

template <typename ParamType>
using DescriptorForPlanTrans = typename CGraphTrans<ParamType>::Descriptor;


/*
 * Implementation for class PlanTrans
 */
template <typename ParamType>
PlanTrans<ParamType>::PlanTrans(
    const std::shared_ptr<ParamType> &param, TensorIdx tune_loop_num,
    TensorIdx tune_para_num, TensorIdx heur_loop_num, TensorIdx heur_para_num,
    GenNumType thread_num, GenNumType tune_times)
    : param_(param),
      optimizer_(param, tune_loop_num, tune_para_num, heur_loop_num,
          heur_para_num, thread_num),
      optimal_descriptor_(this->tuning_(this->optimizer_.get_optimal(),
          tune_times)) {
}


template <typename ParamType>
CGraphTrans<ParamType> *PlanTrans<ParamType>::get_graph() {
  // Return tuned result
  return new CGraphTrans<ParamType>(this->param_, this->optimal_descriptor_);
}


template <typename ParamType>
typename CGraphTrans<ParamType>::Descriptor PlanTrans<ParamType>::tuning_(
    const std::vector<typename CGraphTrans<ParamType>::Descriptor> &descriptors,
    GenNumType tune_times) {
  auto cand_num = static_cast<TensorIdx>(descriptors.size());
  if (1 == cand_num)
    return descriptors[0];

  // Create timer
  TimerWrapper timer(tune_times);

  // Back up coefficients and set identical coefficients for testing
  auto alpha = this->param_->alpha;
  auto beta = this->param_->beta;
  this->param_->set_coef(0.0, 1.0);

  // Measure candidates
  TensorIdx best_idx = 0;
  CGraphTrans<ParamType> candidate(this->param_, descriptors[best_idx]);
  auto best_time = timer(candidate);
  for (TensorIdx cand_idx = best_idx + 1; cand_idx < cand_num; ++cand_idx) {
    candidate.init(descriptors[cand_idx]);
    auto new_time = timer(candidate);
    if (new_time < best_time)
      best_idx = cand_idx, best_time = new_time;
  }

  // Recover coefficients
  this->param_->set_coef(alpha, beta);
  return descriptors[best_idx];
}

#endif // HPTC_PLAN_PLAN_TRANS_TCC_
