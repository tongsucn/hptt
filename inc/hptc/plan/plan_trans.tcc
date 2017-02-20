#pragma once
#ifndef HPTC_PLAN_PLAN_TRANS_TCC_
#define HPTC_PLAN_PLAN_TRANS_TCC_

template <typename ParamType,
          TensorOrder ORDER>
PlanTrans<ParamType, ORDER>::PlanTrans(
    const std::shared_ptr<ParamType> &param, GenNumType thread_num)
    : param_(param),
      optimizer_(param, thread_num) {
}


template <typename ParamType,
          TensorOrder ORDER>
CGraphTrans<ParamType, ORDER> *PlanTrans<ParamType, ORDER>::get_graph(
    TensorIdx num) {
  TensorIdx heur_loop_num, heur_para_num;
  if (num >= 0)
    heur_loop_num = heur_para_num = static_cast<TensorIdx>(std::sqrt(num));
  else
    heur_loop_num = heur_para_num = -1;

  // Construct graph descriptor
  auto descriptor = this->optimizer_.get_optimal(heur_loop_num, heur_para_num);
  return new CGraphTrans<ParamType, ORDER>(this->param_, descriptor);
}

#endif // HPTC_PLAN_PLAN_TRANS_TCC_
