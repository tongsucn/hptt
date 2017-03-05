#pragma once
#ifndef HPTC_PLAN_PLAN_TRANS_UTIL_TCC_
#define HPTC_PLAN_PLAN_TRANS_UTIL_TCC_

template <typename ParamType,
          TensorOrder ORDER>
PlanTransOptimizer<ParamType, ORDER>::PlanTransOptimizer(
    const std::shared_ptr<ParamType> &param, GenNumType thread_num)
    : param_(param->merged_order <= 1 ? nullptr : param),
      threads_(thread_num),
      strategy_(),
      descriptor_(1) {
  if (nullptr == this->param_)
    return;

  this->init_();
}


template <typename ParamType,
          TensorOrder ORDER>
std::vector<CGraphTransDescriptor<ORDER>>
PlanTransOptimizer<ParamType, ORDER>::get_optimal(TensorIdx heur_loop_num,
    TensorIdx heur_para_num, TensorIdx tune_loop_num, TensorIdx tune_para_num) {
  // Check plan status
  if (nullptr == this->param_)
    return {};

  // Heuristics of loop order
  auto loop_orders = this->heur_loop_explorer_(heur_loop_num, tune_loop_num);

  auto cand_num = static_cast<TensorIdx>(loop_orders.size());
  std::vector<CGraphTransDescriptor<ORDER>> result(cand_num, this->descriptor_);
  for (TensorIdx cand_idx = 0; cand_idx < cand_num; ++cand_idx)
    result[cand_idx].loop_order = loop_orders[cand_idx];

  // Heuristics of parallelization
  this->heur_parallel_explorer_(heur_para_num, tune_para_num);

  return result;
}


template <typename ParamType,
          TensorOrder ORDER>
void PlanTransOptimizer<ParamType, ORDER>::init_() {
  // Initialize thread number
  this->init_thread_num_();

  // Check input and output leading and initialize vectorization
  if (this->param_->is_common_leading())
    // Input and output tensor's leading order ARE the same.
    this->init_vec_cl_();
  else
    // Input and output tensor's leading order ARE NOT the same.
    this->init_vec_();

  // Initialize default loop order
  this->init_loop_();

  // Initialize parallelization (expand descriptor)
  this->init_parallel_();
}


template <typename ParamType,
          TensorOrder ORDER>
void PlanTransOptimizer<ParamType, ORDER>::init_thread_num_() {
  // If the input thread number is zero, set thread number to maximum
  if (0 == this->threads_)
    this->threads_ = omp_get_max_threads();
  // If OpenMP returns bad number, set to single thread
  if (this->threads_ <= 0)
    this->threads_ = 1;
}


template <typename ParamType,
          TensorOrder ORDER>
void PlanTransOptimizer<ParamType, ORDER>::init_vec_() {
  // Get parameters
  const auto leadings = this->param_->get_leading();
  auto input_leading = leadings.first;
  auto output_leading = leadings.second;

  // Vectorize single thread version
  auto &oper = this->descriptor_.description[0];
  TensorIdx oper_idx = 0;
  TensorOrder cont_rests[] = { input_leading, input_leading, input_leading };
  TensorOrder ncont_rests[] = {
      output_leading, output_leading, output_leading };

  // Vectorization
  // Full big kernel (4 ncont x 4 cont full macro)
  this->init_vec_kernels_(oper[oper_idx], this->param_->kn_fb.get_cont_len(),
      this->param_->kn_fb.get_ncont_len(), cont_rests[0], ncont_rests[0]);

  // Full vertical kernel (4 ncont x 1 cont full macro)
  ++oper_idx;
  this->init_vec_kernels_(oper[oper_idx], this->param_->kn_fv.get_cont_len(),
      this->param_->kn_fv.get_ncont_len(), cont_rests[0], ncont_rests[1]);
  ncont_rests[0] = ncont_rests[1] = std::min(ncont_rests[0], ncont_rests[1]);

  // Full horizontal kernel (1 ncont x 4 cont full macro)
  ++oper_idx;
  this->init_vec_kernels_(oper[oper_idx], this->param_->kn_fh.get_cont_len(),
      this->param_->kn_fh.get_ncont_len(), cont_rests[1], ncont_rests[0]);

  // Full small kernel (1 ncont x 1 cont full macro)
  ++oper_idx;
  this->init_vec_kernels_(oper[oper_idx], this->param_->kn_fs.get_cont_len(),
      this->param_->kn_fs.get_ncont_len(), cont_rests[1], ncont_rests[1]);
  cont_rests[0] = cont_rests[1] = std::min(cont_rests[0], cont_rests[1]);
  ncont_rests[0] = ncont_rests[1] = std::min(ncont_rests[0], ncont_rests[1]);

  // Half vertical kernel (2 ncont x 1 cont half macro)
  ++oper_idx;
  this->init_vec_kernels_(oper[oper_idx], this->param_->kn_hv.get_cont_len(),
      this->param_->kn_hv.get_ncont_len(), cont_rests[1], ncont_rests[2]);
  ncont_rests[2] = std::min(ncont_rests[0], ncont_rests[2]);

  // Half horizontal kernel (1 ncont x 2 cont half macro)
  ++oper_idx;
  this->init_vec_kernels_(oper[oper_idx], this->param_->kn_hh.get_cont_len(),
      this->param_->kn_hh.get_ncont_len(), cont_rests[2], ncont_rests[1]);
  cont_rests[2] = std::min(cont_rests[0], cont_rests[2]);

  // Half small kernel (1 ncont x 1 cont half macro)
  ++oper_idx;
  this->init_vec_kernels_(oper[oper_idx], this->param_->kn_hs.get_cont_len(),
      this->param_->kn_hs.get_ncont_len(), cont_rests[0], ncont_rests[0]);
  cont_rests[1] = cont_rests[2]
      = std::min(std::min(cont_rests[0], cont_rests[1]), cont_rests[2]);
  ncont_rests[1] = ncont_rests[2]
      = std::min(std::min(ncont_rests[0], ncont_rests[1]), ncont_rests[2]);

  // Scalar vertical
  ++oper_idx;
  this->init_vec_kernels_(oper[oper_idx], 1, 1, cont_rests[2], output_leading,
      true);

  // Scalar horizontal
  ++oper_idx;
  this->init_vec_kernels_(oper[oper_idx], 1, 1, input_leading, ncont_rests[2]);
}


template <typename ParamType,
          TensorOrder ORDER>
void PlanTransOptimizer<ParamType, ORDER>::init_vec_kernels_(
    LoopParam<ORDER> &loop, const GenNumType kn_cont_len,
    const GenNumType kn_ncont_len, TensorOrder &cont_rest,
    TensorOrder &ncont_rest, const bool is_sv) {
  if (kn_cont_len <= cont_rest and kn_ncont_len <= ncont_rest) {
    // Skip merged order
    auto leadings = this->param_->get_leading();
    const auto input_leading = leadings.first;
    const auto output_leading = leadings.second;
    const auto begin_order_idx = ORDER - this->param_->merged_order;
    loop.set_pass(begin_order_idx);

    // Locate loop's positions per memory layout
    auto vec_cont_loop_idx = this->param_->is_col_major ?
        begin_order_idx : ORDER - 1;
    auto vec_ncont_order_idx = this->param_->perm[vec_cont_loop_idx]
        + begin_order_idx;

    // Setup loops
    for (auto order_idx = begin_order_idx; order_idx < ORDER; ++order_idx) {
      if (vec_cont_loop_idx == order_idx) {
        loop.loop_begin[order_idx] = input_leading - cont_rest;
        loop.loop_end[order_idx] = (cont_rest / kn_cont_len) * kn_cont_len
            + loop.loop_begin[order_idx];
        loop.loop_step[order_idx] = kn_cont_len;
      }
      else if (vec_ncont_order_idx == order_idx) {
        loop.loop_begin[order_idx] = output_leading - ncont_rest;
        loop.loop_end[order_idx] = (ncont_rest / kn_ncont_len) * kn_ncont_len
            + loop.loop_begin[order_idx];
        loop.loop_step[order_idx] = kn_ncont_len;

        // Modify end if it's vertical scalar kernel
        if (is_sv)
          loop.loop_end[order_idx] -= ncont_rest;
      }
      else {
        loop.loop_begin[order_idx] = 0;
        loop.loop_end[order_idx]
            = this->param_->input_tensor.get_size()[order_idx];
        loop.loop_step[order_idx] = 1;
      }
    }

    // Update rests
    cont_rest %= kn_cont_len;
    ncont_rest %= kn_ncont_len;
  }
  else
    loop.set_disable();
}


template <typename ParamType,
          TensorOrder ORDER>
void PlanTransOptimizer<ParamType, ORDER>::init_vec_cl_() {
  // Get parameters
  const auto input_leading = this->param_->get_leading().first;
  auto cont_rest = input_leading;
  auto &oper = this->descriptor_.description[0];
  TensorIdx oper_idx = 0;

  // Vectorize single thread version
  // Linear big kernel (8 cont macro kernels)
  std::cout << "lb: " << this->param_->kn_lb.get_cont_len() << std::endl;
  std::cout << "rest: " << cont_rest << std::endl;
  this->init_vec_kernels_cl_(oper[oper_idx++],
      this->param_->kn_lb.get_cont_len(), input_leading, cont_rest);

  // Linear middle kernel (4 cont macro kernels)
  std::cout << "lm: " << this->param_->kn_lm.get_cont_len() << std::endl;
  std::cout << "rest: " << cont_rest << std::endl;
  this->init_vec_kernels_cl_(oper[oper_idx++],
      this->param_->kn_lm.get_cont_len(), input_leading, cont_rest);

  // Linear small kernel (2 cont macro kernels)
  std::cout << "ls: " << this->param_->kn_ls.get_cont_len() << std::endl;
  std::cout << "rest: " << cont_rest << std::endl;
  this->init_vec_kernels_cl_(oper[oper_idx++],
      this->param_->kn_ls.get_cont_len(), input_leading, cont_rest);

  // Linear nano kernel (1 cont macro kernel)
  std::cout << "ln: " << this->param_->kn_ln.get_cont_len() << std::endl;
  std::cout << "rest: " << cont_rest << std::endl;
  this->init_vec_kernels_cl_(oper[oper_idx++],
      this->param_->kn_ln.get_cont_len(), input_leading, cont_rest);

  // Scalar kernel (1 scalar kernel)
  this->init_vec_kernels_cl_(oper[oper_idx], 1, input_leading, cont_rest);
}


template <typename ParamType,
          TensorOrder ORDER>
void PlanTransOptimizer<ParamType, ORDER>::init_vec_kernels_cl_(
    LoopParam<ORDER> &loop, const GenNumType kn_len,
    const TensorOrder input_leading, TensorOrder &cont_rest) {
  if (kn_len <= cont_rest) {
    const auto &begin_idx = this->param_->begin_order_idx;
    const auto ld_idx = this->param_->is_col_major ? begin_idx : ORDER - 1;
    loop.set_pass(begin_idx);

    for (auto loop_idx = begin_idx; loop_idx < ORDER; ++loop_idx) {
      // Initialize vectorized loop
      if (ld_idx != loop_idx) {
        loop.loop_begin[loop_idx] = 0;
        loop.loop_end[loop_idx] = this->param_->input_tensor.get_size()[loop_idx];
        loop.loop_step[loop_idx] = 1;
      }
      else {
        loop.loop_begin[ld_idx] = input_leading - cont_rest;
        loop.loop_end[ld_idx] = (cont_rest / kn_len) * kn_len
          + loop.loop_begin[ld_idx];
        loop.loop_step[ld_idx] = kn_len;
      }
    }

    // Update rests
    cont_rest %= kn_len;
  }
  else
    loop.set_disable();
}


template <typename ParamType,
          TensorOrder ORDER>
void PlanTransOptimizer<ParamType, ORDER>::init_loop_() {
  struct Loop {
    Loop(TensorIdx size, TensorOrder org_idx) : size(size), org_idx(org_idx) {}
    TensorIdx size;
    TensorOrder org_idx;
  };

  // Initialize loop evaluator's parameters
  this->init_loop_evaluator_param_();

  // Locate loop re-order position (first loop that need to be re-ordered)
  const auto begin_idx = this->param_->begin_order_idx;
  const auto end_idx = ORDER - 2 + this->param_->is_common_leading() ? 1 : 0;
  const auto output_ld_idx = this->param_->perm[begin_idx] + begin_idx;

  // Create and initialize loop description array
  std::vector<Loop> loops;
  const auto &size_obj = this->param_->input_tensor.get_size();
  for (TensorOrder loop_idx = 0; loop_idx < ORDER; ++loop_idx)
    if (begin_idx != loop_idx and output_ld_idx != loop_idx)
      loops.push_back(Loop(size_obj[loop_idx], loop_idx));

  loops.push_back(Loop(size_obj[begin_idx], begin_idx));
  if (not this->param_->is_common_leading())
    loops.push_back(Loop(size_obj[output_ld_idx], output_ld_idx));


  // Sort non-leading orders according to the sizes, for loop with smallest size
  // will be put at outer most for loop
  std::sort(loops.begin() + begin_idx, loops.begin() + end_idx,
      [] (const auto &first, const auto &second) -> bool {
        return first.size < second.size; });

  // Set loop order
  for (TensorOrder loop_idx = 0; loop_idx < ORDER; ++loop_idx)
    this->descriptor_.loop_order[loop_idx] = loops[loop_idx].org_idx;
/*
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
  // loops
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
  LoopOrder<ORDER> loop_order;
  for (GenNumType loop_idx = 0; loop_idx < ORDER; ++loop_idx)
    loop_order[loop_idx] = loops[loop_idx].org_idx;
  for (auto loop_idx = begin_idx; loop_idx < end_idx; ++loop_idx)
    strategy.push_back(loops[loop_idx].thread_num);

  // Truncate single threaded inner loops
  TensorIdx trunc_idx = static_cast<TensorIdx>(strategy.size()) - 1;
  trunc_idx = trunc_idx < 0 ? 0 : trunc_idx;
  for (; trunc_idx > 0; --trunc_idx)
    if (strategy[trunc_idx] > 1)
      break;
  strategy.resize(trunc_idx + 1, 1);*/
}


template <typename ParamType,
          TensorOrder ORDER>
void PlanTransOptimizer<ParamType, ORDER>::init_loop_evaluator_param_() {
  this->penalty_begin = 0.0, this->penalty_step = 20.0;
  this->importance_begin = 1.0, this->importance_scale = 0.5;
  this->input_penalty_factor = 1.0, this->output_penalty_factor = 1.01;
}


template <typename ParamType,
          TensorOrder ORDER>
void PlanTransOptimizer<ParamType, ORDER>::init_parallel_() {
}


template <typename ParamType,
          TensorOrder ORDER>
std::vector<LoopOrder<ORDER>>
PlanTransOptimizer<ParamType, ORDER>::heur_loop_explorer_(
    const TensorIdx heur_num, const TensorIdx tune_num) {
  if (0 == heur_num)
    return { this->descriptor_.loop_order };

  // Create best heap to store auto tuning candidates
  // "Cost-Order" pair: (cost, loop order)
  using OrderDes = std::pair<double, LoopOrder<ORDER>>;
  auto heap_cmp = [] (const auto &a, const auto &b) -> bool {
      return a.first < b.first; };

  LoopOrder<ORDER> loop_order;
  const auto ld_idx = this->param_->is_col_major ?
      this->param_->begin_order_idx : ORDER - 1;
  for (TensorOrder order_idx = 0; order_idx < ORDER; ++order_idx)
    loop_order[order_idx] = order_idx;
  std::priority_queue<OrderDes, std::vector<OrderDes>, decltype(heap_cmp)>
      best_heap(heap_cmp);

  // Create initialization loop order
  // Look for best
  bool has_next = true;
  TensorIdx times = 0;
  do {
    // Skip stride-1 leading loop in common leading case
    if (this->param_->is_common_leading() and ld_idx == loop_order[0]) {
      std::next_permutation(
          loop_order.begin() + this->param_->begin_order_idx, loop_order.end());
      continue;
    }

    auto new_cost = this->heur_loop_evaluator_(loop_order);
    if (tune_num < 0 or tune_num > best_heap.size())
      best_heap.push(OrderDes(new_cost, loop_order));
    else
      if (best_heap.top().first > new_cost) {
        best_heap.pop();
        best_heap.push(OrderDes(new_cost, loop_order));
      }
  } while (has_next and (times < heur_num or heur_num < 0));

  // Create result
  std::vector<LoopOrder<ORDER>> result;
  result.reserve(best_heap.size());
  while (not best_heap.empty()) {
    result.push_back(best_heap.top().second);
    best_heap.pop();
  }

  return result;
}


template <typename ParamType,
          TensorOrder ORDER>
double PlanTransOptimizer<ParamType, ORDER>::heur_loop_evaluator_(
    const LoopOrder<ORDER> &target_loop_order) {
  // Locate begin index
  const auto merged_order = this->param_->merged_order;
  const auto begin_idx = this->param_->begin_order_idx;

  // Create loop penalty array
  // [..., 2 * penalty_step, 1 * penalty_step, 0 * penalty_step]
  std::vector<double> loop_penalty(merged_order, 0.0);
  if (this->param_->is_col_major) {
    loop_penalty.back() = this->penalty_begin;
    for (TensorIdx loop_idx = merged_order - 2; loop_idx >= 0; --loop_idx)
      loop_penalty[loop_idx] = loop_penalty[loop_idx + 1] + this->penalty_step;
  }
  else {
    loop_penalty.front() = this->penalty_begin;
    for (TensorIdx loop_idx = 1; loop_idx < merged_order; --loop_idx)
      loop_penalty[loop_idx] = loop_penalty[loop_idx - 1] + this->penalty_step;
  }

  // Create target loop order's index map
  // Key is a specific tensor order, values is its level in loop
  std::vector<TensorOrder> target_map(merged_order, 0);
  for (TensorIdx loop_idx = begin_idx; loop_idx < ORDER; ++loop_idx)
    target_map[target_loop_order[loop_idx] - begin_idx] = loop_idx - begin_idx;

  // Compute target loop order's costs
  double loop_cost = 0.0, importance = importance_begin;
  for (TensorOrder order_idx = 0; order_idx < merged_order;
      ++order_idx, importance *= this->importance_scale) {
    auto input_order_loop_pos = target_map[order_idx];
    auto abs_idx = order_idx + begin_idx;
    auto output_order_loop_pos = target_map[this->param_->perm[abs_idx]];

    auto input_order_penalty = loop_penalty[input_order_loop_pos];
    auto output_order_penalty = loop_penalty[output_order_loop_pos];
    loop_cost += importance * (input_order_penalty * this->input_penalty_factor
        + output_order_penalty * this->output_penalty_factor);
  }

  return loop_cost;
}


template <typename ParamType,
          TensorOrder ORDER>
void PlanTransOptimizer<ParamType, ORDER>::heur_parallel_explorer_(
    const TensorIdx heur_num, const TensorIdx tune_num) {
}


template <typename ParamType,
          TensorOrder ORDER>
LoopOrder<ORDER> PlanTransParallelizer<ParamType, ORDER>::operator()(
    CGraphTransDescriptor<ORDER> &descriptor, GenNumType threads) {
  // Construct result loop order array
  LoopOrder<ORDER> loop_order;
  std::fill(loop_order.begin(), loop_order.end(), 0);

  // Check input descriptor
  if (1 != descriptor.size())
    return loop_order;

  // Set thread number
  if (0 == threads)
    threads = static_cast<GenNumType>(omp_get_max_threads());
  if (0 == threads)
    threads = 1;

  // Resize descriptor according to the thread number
  descriptor.resize(threads, descriptor[0]);

  // Calculate parallelization depth and loop order
  std::vector<GenNumType> strategy;
  loop_order = this->calc_depth_(descriptor, strategy);

  // Parallelize descriptor
  this->parallelize_(descriptor, loop_order, strategy);

  return loop_order;
}


template <typename ParamType,
          TensorOrder ORDER>
void PlanTransParallelizer<ParamType, ORDER>::parallelize_(
    CGraphTransDescriptor<ORDER> &des, const LoopOrder<ORDER> &loop_order,
    const std::vector<GenNumType> &strategy) {
  // This function assumes parallelization strategy is correct,
  // ill-formed strategy will lead to undefined behavior.
  auto threads = std::accumulate(strategy.begin(), strategy.end(), 1,
      std::multiplies<GenNumType>());
  auto max_threads = des.size();
  for (auto idx = threads; idx < max_threads; ++idx)
    des[idx][0].set_disable();
  if (threads <= 1)
    return;

  // Locate begin order for parallelization in case order merge
  const auto begin_idx
      = static_cast<TensorIdx>(ORDER - this->param_->merged_order);
  const auto end_idx
      = begin_idx + static_cast<TensorIdx>(strategy.size());

  // Parallelize
  auto kn_num = static_cast<TensorIdx>(des[0].size());
  for (TensorIdx kn_idx = 0; kn_idx < kn_num; ++kn_idx) {
    // Skip disabled kernel
    auto &kn_basis = des[0][kn_idx];
    if (kn_basis.is_disabled())
      continue;

    auto assign_threads = threads;
    for (auto order_idx = begin_idx; order_idx < end_idx; ++order_idx) {
      // Locate loop
      auto loop_idx = loop_order[order_idx];

      // Assign steps at loop level order_idx to threads
      GenNumType steps =
          (kn_basis.loop_end[loop_idx] - kn_basis.loop_begin[loop_idx]) /
          kn_basis.loop_step[loop_idx];

      // Create split step vector
      const auto curr_para = strategy[order_idx - begin_idx];
      std::vector<TensorIdx> split_steps(curr_para, steps / curr_para);

      // Deal with rest steps
      auto rest_steps = steps % curr_para;
      std::for_each(split_steps.end() - rest_steps, split_steps.end(),
          [] (auto &num) { ++num; });

      // Create unit spans
      std::vector<TensorIdx> unit_begins(assign_threads),
          unit_ends(assign_threads);
      GenNumType copies = assign_threads / curr_para;
      for (GenNumType cp_idx = 0, begin_val = kn_basis.loop_begin[loop_idx];
          cp_idx < curr_para; ++cp_idx) {
        auto cp_beg = cp_idx * copies;
        auto cp_end = cp_beg + copies;
        std::fill(unit_begins.begin() + cp_beg, unit_begins.begin() + cp_end,
            begin_val);
        std::fill(unit_ends.begin() + cp_beg, unit_ends.begin() + cp_end,
            begin_val + split_steps[cp_idx]);
        begin_val += split_steps[cp_idx];
      }

      // Assign rest index in loop to threads
      std::vector<GenNumType> begins(threads), ends(threads);
      for (GenNumType offset = 0; offset < threads;
          offset += assign_threads) {
        std::copy(unit_begins.begin(), unit_begins.end(),
            begins.begin() + offset);
        std::copy(unit_ends.begin(), unit_ends.end(), ends.begin() + offset);
      }

      for (GenNumType oper_idx = 0; oper_idx < threads; ++oper_idx) {
        des[oper_idx][kn_idx].loop_begin[loop_idx] = begins[oper_idx];
        des[oper_idx][kn_idx].loop_end[loop_idx] = ends[oper_idx];
        des[oper_idx][kn_idx].loop_step[loop_idx] = kn_basis.loop_step[loop_idx];
      }

      assign_threads /= curr_para;
    }
  }
}

#endif // HPTC_PLAN_PLAN_TRANS_UTIL_TCC_
