#pragma once
#ifndef HPTC_PLAN_PLAN_TRANS_UTIL_TCC_
#define HPTC_PLAN_PLAN_TRANS_UTIL_TCC_

template <typename ParamType,
          TensorOrder ORDER>
PlanTransOptimizer<ParamType, ORDER>::PlanTransOptimizer(
    const std::shared_ptr<ParamType> &param, GenNumType thread_num)
    : param_(param->merged_order <= 1 ? nullptr : param),
      threads_(thread_num),
      descriptor_(),
      th_fact_map_() {
  if (nullptr == this->param_)
    return;

  this->init_();
}


template <typename ParamType,
          TensorOrder ORDER>
std::vector<CGraphTransDescriptor<ORDER>>
PlanTransOptimizer<ParamType, ORDER>::get_optimal(TensorIdx heur_loop_num,
    TensorIdx heur_para_num, TensorIdx tune_loop_num,
    TensorIdx tune_para_num) const {
  // Check plan status
  if (nullptr == this->param_)
    return {};

  // Heuristics of loop order
  auto loop_orders = this->heur_loop_explorer_(heur_loop_num, tune_loop_num);

  // Heuristics of parallelization
  auto parallel_strategies = this->heur_parallel_explorer_(heur_para_num,
      tune_para_num);

  // Generate candidates
  return this->gen_candidates_(loop_orders, parallel_strategies);
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

  // Integer factorization of thread number
  // Pair's first element is a prime factor, the second is its times
  for (GenNumType num = 2, target = this->threads_; target > 1; ++num) {
    if (0 == target % num) {
      this->th_fact_map_.push_back(std::pair<GenNumType, GenNumType>(num, 0));
      while (0 == target % num) {
        ++this->th_fact_map_.back().second;
        target /= num;
      }
    }
  }
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
  this->init_vec_kernels_cl_(oper[oper_idx++],
      this->param_->kn_lb.get_cont_len(), input_leading, cont_rest);

  // Linear middle kernel (4 cont macro kernels)
  this->init_vec_kernels_cl_(oper[oper_idx++],
      this->param_->kn_lm.get_cont_len(), input_leading, cont_rest);

  // Linear small kernel (2 cont macro kernels)
  this->init_vec_kernels_cl_(oper[oper_idx++],
      this->param_->kn_ls.get_cont_len(), input_leading, cont_rest);

  // Linear nano kernel (1 cont macro kernel)
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
  // Data structure for describing loop, first is loop's size,
  // second is the order a loop stands for
  using Loop = std::pair<TensorIdx, TensorOrder>;

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
      [] (const auto &a, const auto &b) -> bool { return a.first < b.first; });

  // Set loop order
  for (TensorOrder loop_idx = 0; loop_idx < ORDER; ++loop_idx)
    this->descriptor_.loop_order[loop_idx] = loops[loop_idx].second;
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
  // Create data structure storing parallelization information, first is the
  // size of one loop, second is the number of parallelization at this loop.
  const auto input_ld_idx = this->param_->begin_order_idx;
  const auto output_ld_idx = this->param_->perm[input_ld_idx] + input_ld_idx;
  Loop_ loops;
  for (TensorOrder loop_idx = 0; loop_idx < ORDER; ++loop_idx) {
    loops[loop_idx].first = this->param_->input_tensor.get_size()[loop_idx];
    loops[loop_idx].second = 1;
  }

  // Find the largest loop group among different vectorized loop groups
  TensorOrder largest_idx = 0;
  while (this->descriptor_.description[0][largest_idx].is_disabled())
    ++largest_idx;

  // Find the leading dimension's loop position and reset related loops' sizes
  const auto &largest = this->descriptor_.description[0][largest_idx];
  auto span = largest.loop_end[input_ld_idx] - largest.loop_begin[input_ld_idx];
  loops[input_ld_idx].first = span / largest.loop_step[input_ld_idx];
  span = largest.loop_end[output_ld_idx] - largest.loop_begin[output_ld_idx];
  loops[output_ld_idx].first = span / largest.loop_step[output_ld_idx];

  // Parallelize non-leading order loops that can be exact divided by
  // prime factors
  auto fact_map = this->th_fact_map_;
  std::reverse(fact_map.begin(), fact_map.end());
  this->init_parallel_loop_(loops, fact_map, input_ld_idx, output_ld_idx,
      [] (auto a, auto b) -> bool { return 0 == a % b; });

  // Parallelize non-leading order loops that CANNOT be exact divided by
  // prime factors
  this->init_parallel_loop_(loops, fact_map, input_ld_idx, output_ld_idx,
      [] (auto a, auto b) -> bool { return a > b; });

  // Truncate parallelization strategy if necessary
  TensorOrder trunc_idx = ORDER - 1;
  while (trunc_idx > 0 and 1 == loops[trunc_idx].second)
    --trunc_idx;

  // Write parallelization strategy to template descriptor
  this->descriptor_.parallel_strategy.clear();
  for (TensorOrder order_idx = 0; order_idx <= trunc_idx; ++order_idx)
    this->descriptor_.parallel_strategy.push_back(loops[order_idx].second);
}


template <typename ParamType,
          TensorOrder ORDER>
  template <typename Cmp>
void PlanTransOptimizer<ParamType, ORDER>::init_parallel_loop_(
    Loop_ &loops, Factor_ &fact_map, const TensorOrder input_ld_idx,
    const TensorOrder output_ld_idx, Cmp cmp) const {
  for (auto &fact : fact_map) {
    if (fact.second < 1)
      continue;

    for (auto loop_idx = input_ld_idx + 1; loop_idx < ORDER; ++loop_idx) {
      // Skip leading order loop
      if (output_ld_idx == loop_idx)
        continue;
      while (fact.second > 0 and cmp(loops[loop_idx].first, fact.first)) {
        loops[loop_idx].first /= fact.first;
        loops[loop_idx].second *= fact.first;
        --fact.second;
      }
    }

    // Parallelize output leading order loops first
    TensorIdx target_idx = -1;
    if (0 == loops[output_ld_idx].first % fact.first)
      target_idx = static_cast<decltype(target_idx)>(output_ld_idx);
    else if (0 == loops[input_ld_idx].first % fact.first)
      target_idx = static_cast<decltype(target_idx)>(input_ld_idx);

    if (-1 != target_idx)
      while (fact.second > 0 and cmp(loops[target_idx].first, fact.first)) {
        loops[target_idx].first /= fact.first;
        loops[target_idx].second *= fact.first;
        --fact.second;
      }
  }
}


template <typename ParamType,
          TensorOrder ORDER>
std::vector<LoopOrder<ORDER>>
PlanTransOptimizer<ParamType, ORDER>::heur_loop_explorer_(
    const TensorIdx heur_num, const TensorIdx tune_num) const {
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
    const LoopOrder<ORDER> &target_loop_order) const {
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
std::vector<std::vector<GenNumType>>
PlanTransOptimizer<ParamType, ORDER>::heur_parallel_explorer_(
    const TensorIdx heur_num, const TensorIdx tune_num) const {
  if (0 == heur_num)
    return { this->descriptor_.parallel_strategy };
  return { this->descriptor_.parallel_strategy };
}


template <typename ParamType,
          TensorOrder ORDER>
double PlanTransOptimizer<ParamType, ORDER>::heur_parallel_evaluator_(
    const std::vector<GenNumType> &target_para) const {
}


template <typename ParamType,
          TensorOrder ORDER>
std::vector<CGraphTransDescriptor<ORDER>>
PlanTransOptimizer<ParamType, ORDER>::gen_candidates_(
    const std::vector<LoopOrder<ORDER>> &loop_orders,
    const std::vector<std::vector<GenNumType>> &parallel_strategies) const {
  std::vector<CGraphTransDescriptor<ORDER>> candidates;

  // Permute over different loop orders and parallelization strategies
  for (const auto &loop : loop_orders)
    for (const auto &parallel : parallel_strategies) {
      // Create a new candidate from default single threaded descriptor
      candidates.push_back(this->descriptor_);
      candidates.back().loop_order = loop;
      candidates.back().parallel_strategy = parallel;

      // Parallelization
      this->parallelize_(candidates.back());
    }

  return candidates;
}


template <typename ParamType,
          TensorOrder ORDER>
void PlanTransOptimizer<ParamType, ORDER>::parallelize_(
    CGraphTransDescriptor<ORDER> &descriptor) const {
  auto &des = descriptor.description;
  const auto &strategy = descriptor.parallel_strategy;
  const auto begin_loop_idx = this->param_->begin_order_idx;
  const auto end_loop_idx = strategy.size();

  // Calculate actual thread number and resize description
  const auto threads = std::accumulate(strategy.begin(), strategy.end(), 1,
      std::multiplies<GenNumType>());
  if (threads <= 1)
    return;
  des.resize(threads, des[0]);

  // Parallelize
  const auto kn_num = des[0].size();
  for (auto kn_idx = 0; kn_idx < kn_num; ++kn_idx) {
    // Skip disabled kernel
    auto &kn_basis = des[0][kn_idx];
    if (kn_basis.is_disabled())
      continue;

    auto left_threads = threads;
    for (auto loop_idx = begin_loop_idx; loop_idx < end_loop_idx; ++loop_idx) {
      // Compute step times at current loop level
      TensorIdx steps =
          (kn_basis.loop_end[loop_idx] - kn_basis.loop_begin[loop_idx]) /
          kn_basis.loop_step[loop_idx];

      // Create vector to store steps for each thread at current loop level
      const auto curr_para = strategy[loop_idx];
      std::vector<TensorIdx> split_steps(curr_para, steps / curr_para);
      std::for_each(split_steps.end() - steps % curr_para, split_steps.end(),
          [] (auto &num) { ++num; });

      // Create unit spans
      std::vector<TensorIdx> unit_begins(left_threads), unit_ends(left_threads);
      for (TensorIdx cp_idx = 0, begin_val = kn_basis.loop_begin[loop_idx],
          copies = left_threads / curr_para; cp_idx < curr_para; ++cp_idx) {
        auto cp_beg = cp_idx * copies;
        auto cp_end = cp_beg + copies;
        auto end_val
            = begin_val + split_steps[cp_idx] * kn_basis.loop_step[loop_idx];
        std::fill(unit_begins.begin() + cp_beg, unit_begins.begin() + cp_end,
            begin_val);
        std::fill(unit_ends.begin() + cp_beg, unit_ends.begin() + cp_end,
            end_val);
        begin_val = end_val;
      }

      // Assign rest index in loop to threads
      std::vector<TensorIdx> begins(threads), ends(threads);
      for (GenNumType off = 0; off < threads; off += left_threads) {
        std::copy(unit_begins.begin(), unit_begins.end(), begins.begin() + off);
        std::copy(unit_ends.begin(), unit_ends.end(), ends.begin() + off);
      }

      for (GenNumType th_idx = 0; th_idx < threads; ++th_idx) {
        des[th_idx][kn_idx].loop_begin[loop_idx] = begins[th_idx];
        des[th_idx][kn_idx].loop_end[loop_idx] = ends[th_idx];
        des[th_idx][kn_idx].loop_step[loop_idx] = kn_basis.loop_step[loop_idx];
      }

      left_threads /= curr_para;
    }
  }
}

#endif // HPTC_PLAN_PLAN_TRANS_UTIL_TCC_
