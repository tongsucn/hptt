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

  // Locate loop re-order entry (first loop that need to be re-ordered)
  const auto begin_idx
      = static_cast<TensorIdx>(ORDER - this->param_->merged_order);
  const auto end_idx = static_cast<TensorIdx>(ORDER - 2);

  // Create and initialize loop description array
  LoopNode loops[ORDER];
  for (TensorOrder loop_idx = 0; loop_idx < ORDER; ++loop_idx) {
    loops[loop_idx].size = this->param_->input_tensor.get_size()[loop_idx];
    loops[loop_idx].thread_num = 1;
    loops[loop_idx].org_idx = loop_idx;
  }

  // Assign two leading to the deepest loop level
  TensorIdx input_ld_idx, output_ld_idx;
  if (MemLayout::COL_MAJOR == this->param_->MEM_LAYOUT) {
    input_ld_idx = begin_idx;
    output_ld_idx = this->param_->perm[begin_idx] + begin_idx;
  }
  else {
    input_ld_idx = ORDER - 1;
    output_ld_idx = this->param_->perm[ORDER - 1] + begin_idx;
  }

  // Put input leading index at inner most level, and put output leading index
  // at second inner most level
  if (ORDER == 2)
    std::swap(loops[ORDER - 1], loops[ORDER - 2]);
  else if (ORDER - 2 == output_ld_idx)
    std::swap(loops[ORDER - 1], loops[input_ld_idx]);
  else if (ORDER - 1 == output_ld_idx) {
    std::swap(loops[ORDER - 1], loops[ORDER - 2]);
    std::swap(loops[ORDER - 1], loops[input_ld_idx]);
  }
  else {
    std::swap(loops[ORDER - 1], loops[input_ld_idx]);
    std::swap(loops[ORDER - 2], loops[output_ld_idx]);
  }

  // Integer factorization of thread number
  // Pair's first element is a prime factor, the second is its times
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

  // Parallelization
  // Assign prime factors (factorized thread number) to loops
  for (auto loop_idx = begin_idx; loop_idx < end_idx; ++loop_idx) {
    for (auto &fact : fact_map) {
      if (fact.second > 0 and 0 == loops[loop_idx].size % fact.first) {
        loops[loop_idx].size /= fact.first;
        loops[loop_idx].thread_num *= fact.first;
        --fact.second;
      }
    }
  }

  // Sort non-leading loops before parallelize other part, after sorting,
  // loops with larger sizes will be put at inner level (inner most two levels
  // are still the leading loops)
  std::sort(loops + begin_idx, loops + end_idx,
      [] (const auto &first, const auto &second) -> bool {
        return first.size < second.size; });
  // Parallelize with rest threads, begin from inner non-leading loops (larger
  // loops)
  auto fact_map_size = static_cast<TensorIdx>(fact_map.size());
  for (auto fact_idx = fact_map_size - 1; fact_idx >= 0; --fact_idx) {
    for (TensorIdx loop_idx = end_idx - 1;
        fact_map[fact_idx].second > 0 and loop_idx >= begin_idx; --loop_idx) {
      while (fact_map[fact_idx].second > 0 and
          loops[loop_idx].size >= fact_map[fact_idx].first) {
        loops[loop_idx].size /= fact_map[fact_idx].first;
        loops[loop_idx].thread_num *= fact_map[fact_idx].first;
        --fact_map[fact_idx].second;
      }
    }
  }

  // Create loop order and parallelization strategy
  std::array<TensorOrder, ORDER> loop_order;
  for (GenNumType loop_idx = 0; loop_idx < ORDER; ++loop_idx)
    loop_order[loop_idx] = loops[loop_idx].org_idx;
  std::vector<GenNumType> strategy;
  for (auto loop_idx = begin_idx; loop_idx < end_idx; ++loop_idx)
    strategy.push_back(loops[loop_idx].thread_num);

  // Truncate single threaded inner loops
  TensorIdx trunc_idx = static_cast<TensorIdx>(strategy.size()) - 1;
  trunc_idx = trunc_idx < 0 ? 0 : trunc_idx;
  for (; trunc_idx > 0; --trunc_idx)
    if (strategy[trunc_idx] > 1)
      break;
  strategy.resize(trunc_idx + 1, 1);

  std::cout << "Loop order: ";
  for (auto i : loop_order)
    std::cout << i << ", ";
  std::cout << std::endl;

  std::cout << "Strategy: ";
  for (auto i : strategy)
    std::cout << i << ", ";
  std::cout << std::endl;

  return new CGraphTrans<ParamType, ORDER>(this->param_, loop_order, strategy);
}


template <typename ParamType,
          TensorOrder ORDER>
CGraphTrans<ParamType, ORDER> *PlanTrans<ParamType, ORDER>::cgraph_heur_() {
  return nullptr;
}

#endif // HPTC_PLAN_PLAN_TRANS_TCC_
