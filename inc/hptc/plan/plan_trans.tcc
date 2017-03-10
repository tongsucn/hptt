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
    TensorIdx heur_num, TensorIdx tune_num, GenNumType tune_times) {
  // Compute heuristic number
  TensorIdx heur_loop_num, heur_para_num;
  if (heur_num >= 0)
    heur_loop_num = heur_para_num = static_cast<TensorIdx>(std::sqrt(heur_num));
  else
    heur_loop_num = heur_para_num = -1;

  // Compute auto tuning number
  TensorIdx tune_loop_num, tune_para_num;
  if (tune_num >= 0)
    tune_loop_num = tune_para_num = static_cast<TensorIdx>(std::sqrt(tune_num));
  else
    tune_loop_num = tune_para_num = -1;

  // Construct graph descriptor
  auto descriptors = this->optimizer_.get_optimal(heur_loop_num, heur_para_num,
      tune_loop_num, tune_para_num);

  // Return tuned result
  return this->tuning_(descriptors, tune_times);
}


template <typename ParamType,
          TensorOrder ORDER>
CGraphTrans<ParamType, ORDER> *PlanTrans<ParamType, ORDER>::tuning_(
    const std::vector<CGraphTransDescriptor<ORDER>> &descriptors,
    GenNumType tune_times) {
  auto cand_num = static_cast<TensorIdx>(descriptors.size());
  if (0 == cand_num)
    return nullptr;
  else if (1 == cand_num)
    return new Graph(this->param_, descriptors[0]);

  // Create fake data parameter
  auto size_obj = this->param_->input_tensor.get_outer_size();
  std::vector<TensorOrder> size_vec(ORDER);
  for (TensorOrder size_idx = 0; size_idx < ORDER; ++size_idx)
    size_vec[size_idx] = size_obj[size_idx];

  // Creating fake data and fake transpose parameter
  DataWrapper<typename ParamType::FloatType> fake_data(size_vec,
      this->param_->input_tensor.get_data(),
      this->param_->output_tensor.get_data());
  auto fake_param = std::make_shared<ParamType>(*this->param_);
  fake_param->input_tensor.set_data(fake_data.org_in_data);
  fake_param->output_tensor.set_data(fake_data.org_out_data);

  // Create timer
  TimerWrapper timer(tune_times);

  // Create candidates
  std::vector<Graph *> candidates(cand_num, nullptr);
  for (TensorIdx cand_idx = 0; cand_idx < cand_num; ++cand_idx)
    candidates[cand_idx] = new Graph(fake_param, descriptors[cand_idx]);

  // Measure
  TensorIdx best_cand_idx = 0;
  auto best_cand_time = timer(*candidates[0]);
  for (TensorIdx cand_idx = 1; cand_idx < cand_num; ++cand_idx) {
    auto curr_time = timer(*candidates[cand_idx]);
    if (curr_time < best_cand_time)
      best_cand_idx = cand_idx, best_cand_time = curr_time;
  }

  // Release in case memory leak
  for (auto cand_ptr : candidates)
    delete cand_ptr;

  return new Graph(this->param_, descriptors[best_cand_idx]);
}

#endif // HPTC_PLAN_PLAN_TRANS_TCC_
