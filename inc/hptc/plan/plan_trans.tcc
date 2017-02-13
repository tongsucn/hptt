#pragma once
#ifndef HPTC_PLAN_PLAN_TRANS_TCC_
#define HPTC_PLAN_PLAN_TRANS_TCC_

template <typename ParamType,
          TensorOrder ORDER>
PlanTrans<ParamType, ORDER>::PlanTrans(
    const std::shared_ptr<ParamType> &param)
    : param_(param) {
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
  // Get maximum possible concurrency
  auto threads = static_cast<GenNumType>(omp_get_max_threads());
  if (0 == threads)
    threads = 1;

  // Set loop order
  // Create order array
  std::array<TensorOrder, ORDER> loop_order;
  for (TensorOrder idx = 0; idx < ORDER; ++idx)
    loop_order[idx] = idx;

  // Locate loop re-order entry (first loop that need to be re-ordered)
  auto begin_idx = ORDER - this->param_->merged_order;

  // Assign two leading to the deepest loop level
  TensorOrder input_ld_idx, output_ld_idx;
  if (MemLayout::COL_MAJOR == this->param_->MEM_LAYOUT) {
    input_ld_idx = begin_idx;
    output_ld_idx = this->param_->perm[begin_idx] + begin_idx;
  }
  else {
    input_ld_idx = ORDER - 1;
    output_ld_idx = this->param_->perm[ORDER - 1] + begin_idx;
  }
  //!< For now, put input leading's index at inner most level
  loop_order[ORDER - 1] = input_ld_idx;
  loop_order[ORDER - 2] = output_ld_idx;

  // Process other loops and get their sizes
  GenNumType other_num = ORDER - begin_idx - 2;
  std::vector<TensorIdx> sizes(other_num);
  for (TensorOrder idx = begin_idx, write_idx = begin_idx; idx < ORDER; ++idx)
    if (idx != input_ld_idx and idx != output_ld_idx) {
      sizes[write_idx - begin_idx] = this->param_->input_tensor.get_size()[idx];
      loop_order[write_idx++] = idx;
    }

  // Depose thread number
  using KV = std::pair<GenNumType, GenNumType>;
  std::vector<KV> fact_map;
  for (GenNumType num = 2, target = threads; target > 1; ++num) {
    if (0 == target % num) {
      fact_map.push_back(KV(num, 0));
      while (0 == target % num) {
        ++fact_map.back().second;
        target /= num;
      }
    }
  }

  // Create parallelization strategy
  // Assign according to factorization
  std::vector<GenNumType> strategy(other_num, 1);
  for (GenNumType idx = 0; idx < other_num; ++idx) {
    for (KV &p : fact_map) {
      if (p.second > 0 and 0 == sizes[idx] % p.first) {
        sizes[idx] /= p.first;
        strategy[idx] *= p.first;
        --p.second;
      }
    }
  }

  // Assign by force
  auto fact_map_size = static_cast<TensorIdx>(fact_map.size());
  for (auto fact_idx = fact_map_size - 1; fact_idx >= 0; --fact_idx) {
    for (TensorIdx idx = 0; fact_map[fact_idx].second > 0 and idx < other_num;
        ++idx) {
      while (fact_map[fact_idx].second > 0
          and sizes[idx] >= fact_map[fact_idx].first) {
        strategy[idx] *= fact_map[fact_idx].first;
        sizes[idx] /= fact_map[fact_idx].first;
        --fact_map[fact_idx].second;
      }
    }
  }

  // Truncate single threads
  TensorIdx trunc_idx = other_num - 1;
  for (; trunc_idx > 0; --trunc_idx)
    if (strategy[trunc_idx] > 1)
      break;
  strategy.resize(trunc_idx + 1);

  return new CGraphTrans<ParamType, ORDER>(this->param_, loop_order, strategy);
}


template <typename ParamType,
          TensorOrder ORDER>
CGraphTrans<ParamType, ORDER> *PlanTrans<ParamType, ORDER>::cgraph_heur_() {
  return nullptr;
}

#endif // HPTC_PLAN_PLAN_TRANS_TCC_
