#pragma once
#ifndef HPTC_PLAN_PLAN_TRANS_TCC_
#define HPTC_PLAN_PLAN_TRANS_TCC_

template <typename ParamType,
          TensorOrder ORDER>
PlanTrans<ParamType, ORDER>::PlanTrans(
    const std::shared_ptr<ParamType> &param)
    : param_(param),
      vectorizer_(param),
      parallelizer_(param) {
}


template <typename ParamType,
          TensorOrder ORDER>
CGraphTrans<ParamType, ORDER> *PlanTrans<ParamType, ORDER>::get_graph(
    PlanTypeTrans plan_type) {
  if (PLAN_TRANS_AUTO == plan_type)
    return this->cgraph_auto_();
  else
    return this->cgraph_heur_();
}


template <typename ParamType,
          TensorOrder ORDER>
CGraphTrans<ParamType, ORDER> *PlanTrans<ParamType, ORDER>::cgraph_auto_() {
  auto descriptor = this->vectorizer_();
  auto loop_order = this->parallelizer_(descriptor);
  return new CGraphTrans<ParamType, ORDER>(this->param_, loop_order,
      descriptor);
}


template <typename ParamType,
          TensorOrder ORDER>
CGraphTrans<ParamType, ORDER> *PlanTrans<ParamType, ORDER>::cgraph_heur_() {
  return nullptr;
}

#endif // HPTC_PLAN_PLAN_TRANS_TCC_
