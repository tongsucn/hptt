#pragma once
#ifndef HPTC_PLAN_PLAN_TRANS_UTIL_TCC_
#define HPTC_PLAN_PLAN_TRANS_UTIL_TCC_

template <typename ParamType,
          TensorOrder ORDER>
PlanTransOptimizer<ParamType, ORDER>::PlanTransOptimizer(
    const std::shared_ptr<ParamType> &param, GenNumType thread_num)
    : param_(param),
      threads_(thread_num),
      strategy_(),
      descriptor_(1) {
  this->init_();
}


template <typename ParamType,
          TensorOrder ORDER>
std::vector<CGraphTransDescriptor<ORDER>>
PlanTransOptimizer<ParamType, ORDER>::get_optimal(TensorIdx heur_loop_num,
    TensorIdx heur_para_num, TensorIdx tune_loop_num, TensorIdx tune_para_num) {
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

  // Initialize vectorization
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
  // Check permutation type
  if (-1 == this->param_->perm_type()) {
    // For now, do nothing when leading dimensions are the same.
    return;
  }

  // Get parameters
  auto input_leading = this->param_->input_tensor.get_leading();
  auto output_leading = this->param_->output_tensor.get_leading();

  // Vectorize single thread version
  auto &oper = this->descriptor_.description[0];
  TensorIdx oper_idx = 0;

  // Vectorization
  // Full big kernel (4 ncont x 4 cont full macro)
  auto fb_size = this->param_->kn_fb.get_cont_len();
  TensorOrder fb_cont_rest = input_leading, fb_ncont_rest = output_leading;
  TensorIdx fb_cont_begin = 0, fb_ncont_begin = 0;
  auto fb_set = this->init_vec_kernels_(oper[oper_idx], fb_size, fb_size,
      fb_cont_rest, fb_ncont_rest, fb_cont_begin, fb_ncont_begin);

  // Full vertical kernel (4 ncont x 1 cont full macro)
  ++oper_idx;
  TensorOrder fv_cont_rest = fb_cont_rest, fv_ncont_rest = output_leading;
  TensorIdx fv_cont_begin = fb_cont_begin, fv_ncont_begin = 0;
  auto fv_set = this->init_vec_kernels_(oper[oper_idx],
      this->param_->kn_fv.get_cont_len(), this->param_->kn_fv.get_ncont_len(),
      fv_cont_rest, fv_ncont_rest, fv_cont_begin, fv_ncont_begin);

  // Full horizontal kernel (1 ncont x 4 cont full macro)
  ++oper_idx;
  TensorOrder fh_cont_rest = input_leading, fh_ncont_rest = fb_ncont_rest;
  TensorIdx fh_cont_begin = 0, fh_ncont_begin = fb_ncont_begin;
  auto fh_set = this->init_vec_kernels_(oper[oper_idx],
      this->param_->kn_fh.get_cont_len(), this->param_->kn_fh.get_ncont_len(),
      fh_cont_rest, fh_ncont_rest, fh_cont_begin, fh_ncont_begin);

  // Full small kernel (1 ncont x 1 cont full macro)
  ++oper_idx;
  auto fs_size = this->param_->kn_fs.get_cont_len();
  TensorOrder fs_cont_rest = input_leading, fs_ncont_rest = output_leading;
  TensorIdx fs_cont_begin = 0, fs_ncont_begin = 0;
  if (fb_set) {
    fs_cont_rest = fb_cont_rest, fs_ncont_rest = fb_ncont_rest;
    fs_cont_begin = fb_cont_begin, fs_ncont_begin = fb_ncont_begin;
  }
  else if (fv_set)
    fs_ncont_rest = fv_ncont_rest, fs_ncont_begin = fv_ncont_begin;
  else if (fh_set)
    fs_cont_rest = fh_cont_rest, fs_cont_begin = fh_cont_begin;
  auto fs_set = this->init_vec_kernels_(oper[oper_idx], fs_size, fs_size,
      fs_cont_rest, fs_ncont_rest, fs_cont_begin, fs_ncont_begin);

  // Half vertical kernel (2 ncont x 1 cont half macro) and half horizontal
  // kernel (1 ncont x 2 cont half macro)
  ++oper_idx;
  TensorOrder hv_cont_rest = input_leading, hv_ncont_rest = output_leading;
  TensorIdx hv_cont_begin = 0, hv_ncont_begin = 0;
  TensorOrder hh_cont_rest = input_leading, hh_ncont_rest = output_leading;
  TensorIdx hh_cont_begin = 0, hh_ncont_begin = 0;
  TensorOrder hs_cont_rest = input_leading, hs_ncont_rest = output_leading;
  TensorIdx hs_cont_begin = 0, hs_ncont_begin = 0;
  if (fs_set) {
    hv_cont_rest = fs_cont_rest, hh_ncont_rest = fs_ncont_rest;
    hv_cont_begin = fs_cont_begin, hh_ncont_begin = fs_ncont_begin;
    hs_cont_rest = fs_cont_rest, hs_ncont_rest = fs_ncont_rest;
    hs_cont_begin = fs_cont_begin, hs_ncont_begin = fs_ncont_begin;
  }
  else if (fv_set) {
    hv_cont_rest = fv_cont_rest, hh_ncont_rest = fv_ncont_rest;
    hv_cont_begin = fv_cont_begin, hh_ncont_begin = fv_ncont_begin;
    hs_cont_rest = fv_cont_rest, hs_ncont_rest = fv_ncont_rest;
    hs_cont_begin = fv_cont_begin, hs_ncont_begin = fv_ncont_begin;
  }
  else if (fh_set) {
    hv_cont_rest = fh_cont_rest, hh_ncont_rest = fh_ncont_rest;
    hv_cont_begin = fh_cont_begin, hh_ncont_begin = fh_ncont_begin;
    hs_cont_rest = fh_cont_rest, hs_ncont_rest = fh_ncont_rest;
    hs_cont_begin = fh_cont_begin, hs_ncont_begin = fh_ncont_begin;
  }
  else if (fb_set) {
    hv_cont_rest = fb_cont_rest, hh_ncont_rest = fb_ncont_rest;
    hv_cont_begin = fb_cont_begin, hh_ncont_begin = fb_ncont_begin;
    hs_cont_rest = fb_cont_rest, hs_ncont_rest = fb_ncont_rest;
    hs_cont_begin = fb_cont_begin, hs_ncont_begin = fb_ncont_begin;
  }
  auto hv_set = this->init_vec_kernels_(oper[oper_idx],
      this->param_->kn_hv.get_cont_len(), this->param_->kn_hv.get_ncont_len(),
      hv_cont_rest, hv_ncont_rest, hv_cont_begin, hv_ncont_begin);

  ++oper_idx;
  auto hh_set = this->init_vec_kernels_(oper[oper_idx],
      this->param_->kn_hh.get_cont_len(), this->param_->kn_hh.get_ncont_len(),
      hh_cont_rest, hh_ncont_rest, hh_cont_begin, hh_ncont_begin);

  // Half small kernel (1 ncont x 1 cont half macro)
  ++oper_idx;
  auto hs_size = this->param_->kn_hs.get_cont_len();
  if (hv_set and hh_set)
    ;
  else if (hv_set)
    hs_ncont_begin = hv_ncont_begin, hs_ncont_rest = hv_ncont_rest;
  else if (hh_set)
    hs_ncont_begin = hh_ncont_begin, hs_ncont_rest = hh_ncont_rest;
  auto hs_set = this->init_vec_kernels_(oper[oper_idx], hs_size, hs_size,
      hs_cont_rest, hs_ncont_rest, hs_cont_begin, hs_ncont_begin);

  // Scalar horizontal and then scalar vertical
  ++oper_idx;
  TensorOrder sh_cont_rest = input_leading, sh_ncont_rest = hs_ncont_rest;
  TensorIdx sh_cont_begin = 0, sh_ncont_begin = hs_ncont_begin;
  TensorOrder sv_cont_rest = hs_cont_rest, sv_ncont_rest = sh_ncont_begin;
  TensorIdx sv_cont_begin = hs_cont_begin, sv_ncont_begin = 0;
  if (hs_set)
    ;
  else if (hv_set)
    sv_cont_rest = hv_cont_rest, sv_cont_begin = hv_cont_begin;
  else if (hh_set) {
    sh_ncont_rest = hh_ncont_rest, sh_ncont_begin = hh_ncont_begin;
    sv_ncont_rest = sh_ncont_begin;
  }
  this->init_vec_kernels_(oper[oper_idx], 1, 1, sh_cont_rest, sh_ncont_rest,
      sh_cont_begin, sh_ncont_begin);

  ++oper_idx;
  this->init_vec_kernels_(oper[oper_idx], 1, 1, sv_cont_rest, sv_ncont_rest,
      sv_cont_begin, sv_ncont_begin);
}


template <typename ParamType,
          TensorOrder ORDER>
bool PlanTransOptimizer<ParamType, ORDER>::init_vec_kernels_(
    LoopParam<ORDER> &loop, GenNumType cont_len, GenNumType ncont_len,
    TensorOrder &cont_rest, TensorOrder &ncont_rest,
    TensorIdx &cont_begin, TensorIdx &ncont_begin) {
  if (cont_len <= cont_rest and ncont_len <= ncont_rest) {
    const TensorIdx begin_idx = ORDER - this->param_->merged_order;
    const TensorIdx ncont_rest_idx = begin_idx + this->param_->perm[begin_idx];

    // Set pass on merged orders
    loop.set_pass(static_cast<TensorOrder>(begin_idx));
    // Vectorize
    for (TensorIdx idx = begin_idx; idx < ORDER; ++idx) {
      if (begin_idx == idx) {
        // Vectorize input tensor stride-1 order
        TensorIdx times = cont_rest / cont_len;
        loop.loop_begin[idx] = cont_begin;

        auto span = times * cont_len;
        cont_begin += span;
        cont_rest -= span;

        loop.loop_end[idx] = cont_begin;
        loop.loop_step[idx] = cont_len;
      }
      else if (ncont_rest_idx == idx) {
        // Vectorize output tensor stride-1 order
        TensorIdx times = ncont_rest / ncont_len;
        loop.loop_begin[idx] = ncont_begin;

        auto span = times * ncont_len;
        ncont_begin += span;
        ncont_rest -= span;

        loop.loop_end[idx] = ncont_begin;
        loop.loop_step[idx] = ncont_len;
      }
      else {
        loop.loop_begin[idx] = 0;
        loop.loop_end[idx] = this->param_->input_tensor.get_size()[idx];
        loop.loop_step[idx] = 1;
      }
    }
    return true;
  }
  return false;
}


template <typename ParamType,
          TensorOrder ORDER>
void PlanTransOptimizer<ParamType, ORDER>::init_loop_() {
  // Initialize loop evaluator's parameters
  this->init_loop_evaluator_param_();

  // Locate loop re-order entry (first loop that need to be re-ordered)
  const auto begin_idx
      = static_cast<TensorIdx>(ORDER - this->param_->merged_order);
  const auto end_idx = static_cast<TensorIdx>(ORDER - 2);

  // Create and initialize loop description array
  Loop_ loops[ORDER];
  for (TensorOrder loop_idx = 0; loop_idx < ORDER; ++loop_idx) {
    loops[loop_idx].size = this->param_->input_tensor.get_size()[loop_idx];
    loops[loop_idx].thread_num = 1;
    loops[loop_idx].org_idx = loop_idx;
  }

  // Put input leading index at inner most level, and put output leading index
  // at second inner most level
  TensorIdx input_ld_idx, output_ld_idx;
  if (MemLayout::COL_MAJOR == this->param_->MEM_LAYOUT) {
    input_ld_idx = begin_idx;
    output_ld_idx = this->param_->perm[begin_idx] + begin_idx;
  }
  else {
    input_ld_idx = ORDER - 1;
    output_ld_idx = this->param_->perm[ORDER - 1] + begin_idx;
  }

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

  // Sort non-leading orders according to the sizes
  std::sort(loops + begin_idx, loops + end_idx,
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
    return std::vector<LoopOrder<ORDER>>{ this->descriptor_.loop_order };

  // Create best heap to store auto tuning candidates
  // "Cost-Order" pair
  using OrderDes = std::pair<double, LoopOrder<ORDER>>;
  const auto &best_order = this->descriptor_.loop_order;
  OrderDes curr_best(this->heur_loop_evaluator_(best_order), best_order);
  std::vector<OrderDes> best_heap{ curr_best };

  auto heap_cmp = [] (const auto &a, const auto &b) -> bool {
      return a.first < b.first; };
  std::make_heap(best_heap.begin(), best_heap.end(), heap_cmp);

  // Create initialization loop order
  LoopOrder<ORDER> loop_order;
  for (TensorOrder order_idx = 0; order_idx < ORDER; ++order_idx)
    loop_order[order_idx] = order_idx;

  // Look for best
  bool has_next = true;
  TensorOrder begin_idx = ORDER - this->param_->merged_order;
  auto perm_start = loop_order.begin() + begin_idx;
  for (TensorIdx times = 0; has_next and (heur_num < 0 or times < heur_num);
      ++times, has_next = std::next_permutation(perm_start, loop_order.end())) {
    auto new_cost = this->heur_loop_evaluator_(loop_order);
    if (tune_num < 0) {
      best_heap.push_back(OrderDes(new_cost, loop_order));
      std::push_heap(best_heap.begin(), best_heap.end(), heap_cmp);
    }
    else if (tune_num <= 1) {
      if (best_heap[0].first > new_cost)
        best_heap[0] = OrderDes(new_cost, loop_order);
    }
    else if (tune_num > best_heap.size()) {
      best_heap.push_back(OrderDes(new_cost, loop_order));
      std::push_heap(best_heap.begin(), best_heap.end(), heap_cmp);
    }
    else {
      if (best_heap[0].first > new_cost) {
        std::pop_heap(best_heap.begin(), best_heap.end(), heap_cmp);
        best_heap.back() = OrderDes(new_cost, loop_order);
        std::push_heap(best_heap.begin(), best_heap.end(), heap_cmp);
      }
    }
  }

  // Create result
  std::vector<LoopOrder<ORDER>> result;
  result.reserve(best_heap.size());
  for (auto &des : best_heap)
    result.push_back(des.second);

  return result;
}


template <typename ParamType,
          TensorOrder ORDER>
double PlanTransOptimizer<ParamType, ORDER>::heur_loop_evaluator_(
    const LoopOrder<ORDER> &target_loop_order) {
  // Locate begin index
  auto merged_order = this->param_->merged_order;
  auto begin_idx = ORDER - merged_order;

  // Create loop penalty array
  // [..., 2 * penalty_step, 1 * penalty_step, 0 * penalty_step]
  std::vector<double> loop_penalty(merged_order, 0.0);
  loop_penalty.back() = this->penalty_begin;
  for (TensorIdx loop_idx = merged_order - 1; loop_idx >= 0; --loop_idx)
    loop_penalty[loop_idx] = loop_penalty[loop_idx + 1] + this->penalty_step;

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
