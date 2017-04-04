#pragma once
#ifndef HPTC_PLAN_PLAN_TRANS_UTIL_TCC_
#define HPTC_PLAN_PLAN_TRANS_UTIL_TCC_

/*
 * Implementation for class PlanTransOptimizer
 */
template <typename ParamType>
PlanTransOptimizer<ParamType>::PlanTransOptimizer(
    const std::shared_ptr<ParamType> &param, const TensorUInt num_threads,
    const TensorInt tune_loop_num, const TensorInt tune_para_num,
    const TensorInt heur_loop_num, const TensorInt heur_para_num)
    : param_(param->merged_order <= 1 ? nullptr : param),
      threads_(num_threads),
      in_ld_idx_(this->param_->begin_order_idx),
      out_ld_idx_(this->param_->perm[this->in_ld_idx_] + this->in_ld_idx_),
      th_factor_map_(), avail_parallel_(),
      loop_order_candidates_(), parallel_strategy_candidates_(),
      template_descriptor_() {
  if (nullptr == this->param_)
    return;

  this->init_(tune_loop_num, tune_para_num, heur_loop_num, heur_para_num);
}


template <typename ParamType>
std::vector<typename CGraphTrans<ParamType>::Descriptor>
PlanTransOptimizer<ParamType>::get_optimal() const {
  // Check if parameter is nullptr
  if (nullptr == this->param_)
    return {};
  return this->gen_candidates_();
}


template <typename ParamType>
void PlanTransOptimizer<ParamType>::init_(TensorInt tune_loop_num,
    TensorInt tune_para_num, TensorInt heur_loop_num, TensorInt heur_para_num) {
  // Initialize all kinds of parameters and configurations
  this->init_config_();

  // Initialize loop order
  if (0 == heur_loop_num)
    // Use default strategy
    this->init_loop_rule_();
  else {
    // Use heuristic strategy
    tune_loop_num = 0 == tune_loop_num ? 1 : tune_loop_num;
    if (this->param_->is_common_leading())
      this->init_loop_heur_common_leading_(tune_loop_num, heur_loop_num);
    else
      this->init_loop_heur_general_(tune_loop_num, heur_loop_num);
  }

  // Initialize thread number
  this->init_threads_();

  // Check input and output leading and initialize vectorization
  if (this->param_->is_common_leading())
    this->init_vec_common_leading_();
  else
    this->init_vec_general_();

  // Initialize parallelization strategy
  // Use heuristic strategy
  if (0 != heur_para_num) {
    tune_para_num = 0 == tune_para_num ? 1 : tune_para_num;
    this->init_parallel_heur_(tune_para_num, heur_para_num);
  }

  // Use default strategy or heuristic of parallelization failed
  if (0 == heur_para_num or this->parallel_strategy_candidates_.empty()) {
    if (this->param_->is_common_leading())
      this->init_parallel_rule_common_leading_();
    else
      this->init_parallel_rule_general_();

    // Push default strategy in candidate list
    this->parallel_strategy_candidates_.emplace_back(
        this->template_descriptor_.parallel_strategy);
  }
}


template <typename ParamType>
void PlanTransOptimizer<ParamType>::init_config_() {
  // Initialize available parallelism at every loop with input tensor size.
  // In common leading case, leading's available parallelism will be 1.
  this->avail_parallel_.fill(1);
  for (auto loop_idx = this->in_ld_idx_
          + (this->param_->is_common_leading() ? 1 : 0); loop_idx < ORDER;
      ++loop_idx)
    this->avail_parallel_[loop_idx] = this->param_->input_tensor.get_size(
        loop_idx);

  // Initialize loop evaluator's parameters
  this->init_loop_evaluator_param_();

  // Initialize parallelization evaluator's parameters
  this->init_parallel_evaluator_param_();
}


template <typename ParamType>
void PlanTransOptimizer<ParamType>::init_loop_evaluator_param_() {
  this->heur_loop_penalty_begin = 0.0;
  this->heur_loop_penalty_step = 20.0;
  this->heur_loop_importance_begin = 1.0;
  this->heur_loop_importance_scale = 0.5;
  this->heur_loop_input_penalty_factor = 1.0;
  this->heur_loop_output_penalty_factor = 1.01;
  this->heur_loop_in_ld_award = 0.85;
  this->heur_loop_out_ld_award = 0.8;
}


template <typename ParamType>
void PlanTransOptimizer<ParamType>::init_parallel_evaluator_param_() {
  this->heur_para_penalty_factor_cl = 1.01;
  this->heur_para_penalty_factor_inld = 1.00010;
  this->heur_para_penalty_factor_outld = 1.00015;
  this->heur_para_max_penalty_threads = 16;
  this->heur_para_cost_begin = 1.0;
}


template <typename ParamType>
void PlanTransOptimizer<ParamType>::init_loop_rule_() {
  // Data structure for describing loop, first is a loop's index,
  // second is loop's score
  using LoopScore = std::pair<TensorUInt, double>;

  // Create and initialize loop description array
  std::vector<LoopScore> scores(ORDER, LoopScore(0, 0.0));
  for (auto loop_idx = this->in_ld_idx_; loop_idx < ORDER; ++loop_idx) {
    auto curr_score = static_cast<double>(loop_idx - this->in_ld_idx_);
    scores[loop_idx].first = loop_idx;
    scores[loop_idx].second += curr_score;
    auto perm_idx = this->param_->perm[loop_idx] + this->in_ld_idx_;
    scores[perm_idx].second += curr_score;
  }
  scores[this->in_ld_idx_].second *= this->heur_loop_in_ld_award;
  scores[this->out_ld_idx_].second *= this->heur_loop_out_ld_award;

  // Sort orders according to the scores, if two orders have same score, then
  // they will be sorted by their available parallelism. The loop with lowest
  // scores tend to be placed at inner most loop.
  std::sort(scores.begin() + this->in_ld_idx_, scores.end(),
      [this] (const LoopScore &a, const LoopScore &b) -> bool {
          return a.second > b.second or (a.second == b.second and
          this->avail_parallel_[a.first] < this->avail_parallel_[b.first]); });

  // Set loop order
  this->loop_order_candidates_.emplace_back();
  for (TensorUInt loop_idx = 0; loop_idx < ORDER; ++loop_idx)
    this->loop_order_candidates_.back()[loop_idx] = scores[loop_idx].first;
}


template <typename ParamType>
void PlanTransOptimizer<ParamType>::init_loop_heur_common_leading_(
    const TensorInt tune_num, const TensorInt heur_num) {
  // Create best heap to store auto tuning candidates
  // "Cost-Order" pair: (cost, loop order)
  using OrderDes = std::pair<double, LoopOrderTrans<ORDER>>;
  auto heap_cmp = [] (const OrderDes &a, const OrderDes &b) -> bool {
      return a.first < b.first; };

  // Create initial loop order
  LoopOrderTrans<ORDER> loop_order;
  for (TensorUInt order_idx = 0; order_idx < ORDER; ++order_idx)
    loop_order[order_idx] = order_idx;
  std::priority_queue<OrderDes, std::vector<OrderDes>, decltype(heap_cmp)>
      best_heap(heap_cmp);

  // Skip stride-1 leading loop in common leading case
  std::swap(loop_order[this->in_ld_idx_], loop_order[this->in_ld_idx_ + 1]);

  // Look for best
  TensorInt times = 0;
  for (bool has_next = true; has_next and (heur_num < 0 or heur_num > times);
      ++times, has_next = std::next_permutation(
          loop_order.begin() + this->in_ld_idx_, loop_order.end())) {
    // Update best heap
    auto new_cost = this->heur_loop_evaluator_(loop_order);
    if (tune_num < 0 or tune_num > static_cast<TensorInt>(best_heap.size()))
      best_heap.emplace(new_cost, loop_order);
    else if (best_heap.top().first > new_cost) {
        best_heap.pop();
        best_heap.emplace(new_cost, loop_order);
    }
  }

  // Store result
  while (not best_heap.empty()) {
    this->loop_order_candidates_.emplace_back(best_heap.top().second);
    best_heap.pop();
  }
}


template <typename ParamType>
void PlanTransOptimizer<ParamType>::init_loop_heur_general_(
    const TensorInt tune_num, const TensorInt heur_num) {
  // Create best heap to store auto tuning candidates
  // "Cost-Order" pair: (cost, loop order)
  using OrderDes = std::pair<double, LoopOrderTrans<ORDER>>;
  auto heap_cmp = [] (const OrderDes &a, const OrderDes &b) -> bool {
      return a.first < b.first; };

  // Create initial loop order
  LoopOrderTrans<ORDER> loop_order;
  for (TensorUInt order_idx = 0; order_idx < ORDER; ++order_idx)
    loop_order[order_idx] = order_idx;
  std::priority_queue<OrderDes, std::vector<OrderDes>, decltype(heap_cmp)>
      best_heap(heap_cmp);

  // 1. Force the leading orders' loops to inner most
  std::swap(loop_order[this->out_ld_idx_], loop_order[ORDER - 1]);
  std::swap(loop_order[this->in_ld_idx_], loop_order[ORDER - 2]);

  // Look for best
  TensorInt times = 0;
  for (bool has_next = true; has_next and (heur_num < 0 or heur_num > times);) {
    for (auto swap_count = 2; swap_count > 0;
        --swap_count, std::swap(loop_order[ORDER - 1], loop_order[ORDER - 2])) {
      auto new_cost = this->heur_loop_evaluator_(loop_order);
      if (tune_num < 0 or tune_num > static_cast<TensorInt>(best_heap.size()))
        best_heap.emplace(new_cost, loop_order);
      else if (best_heap.top().first > new_cost) {
          best_heap.pop();
          best_heap.emplace(new_cost, loop_order);
      }
    }

    times += 2;
    has_next = std::next_permutation(loop_order.begin() + this->in_ld_idx_,
        loop_order.end() - 2);
  }

  // Re-initialize loop order descriptor, and skip cases that leading orders
  // locating at outer most loops (when ORDER > 3)
  if (ORDER > 3) {
    for (TensorUInt order_idx = 0; order_idx < ORDER; ++order_idx)
      loop_order[order_idx] = order_idx;
    if (this->in_ld_idx_ + 1 == this->out_ld_idx_)
      std::rotate(loop_order.begin() + this->in_ld_idx_,
          loop_order.begin() + this->in_ld_idx_ + 2,
          loop_order.begin() + this->in_ld_idx_ + 3);
    else
      std::swap(loop_order[this->in_ld_idx_], loop_order[this->in_ld_idx_ + 1]);
  }

  // 2. Allow leading orders appear anywhere except outer most (when ORDER > 3)
  for (bool has_next = true;
      ORDER > 2 and has_next and (heur_num < 0 or heur_num > times);) {
    if (ORDER > 3) {
      if (this->out_ld_idx_ == loop_order[this->in_ld_idx_]) {
        if (ORDER - 1 == this->out_ld_idx_)
          break;
        else
          std::swap(loop_order[this->in_ld_idx_],
              loop_order[this->out_ld_idx_ + 1]);
      }
      else {
        while ((this->in_ld_idx_ == loop_order[ORDER - 1] and
                this->out_ld_idx_ == loop_order[ORDER - 2]) or
            (this->in_ld_idx_ == loop_order[ORDER - 2] and
                this->out_ld_idx_ == loop_order[ORDER - 1]))
          has_next = std::next_permutation(
              loop_order.begin() + this->in_ld_idx_, loop_order.end());
      }

      if (not has_next)
        break;
    }

    auto new_cost = this->heur_loop_evaluator_(loop_order);
    if (tune_num < 0 or tune_num > static_cast<TensorInt>(best_heap.size()))
      best_heap.emplace(new_cost, loop_order);
    else if (best_heap.top().first > new_cost) {
        best_heap.pop();
        best_heap.emplace(new_cost, loop_order);
    }

    ++times;
    has_next = std::next_permutation(loop_order.begin() + this->in_ld_idx_,
        loop_order.end());
  }

  // Store result
  while (not best_heap.empty()) {
    this->loop_order_candidates_.emplace_back(best_heap.top().second);
    best_heap.pop();
  }
}


template <typename ParamType>
void PlanTransOptimizer<ParamType>::init_threads_() {
  // If the input thread number is zero, set thread number according to OpenMP's
  // available thread number
  auto omp_threads = static_cast<TensorInt>(this->threads_);
  if (0 == this->threads_)
    omp_threads = omp_get_max_threads();

  // If OpenMP returns bad number, set to single thread
  if (omp_threads <= 0)
    this->threads_ = 1;
  else
    this->threads_ = omp_threads;

  // Integer factorization of thread number
  this->th_factor_map_ = hptc::factorize(this->threads_);
  this->template_descriptor_.parallel_strategy.fill(1);

  // Parallel non-leading order loops that can be exactly divided, begin from
  // outer most loop.
  const auto out_2nd = this->param_->perm[this->in_ld_idx_ + 1]
      + this->in_ld_idx_;
  for (auto loop_idx = this->in_ld_idx_; loop_idx < ORDER; ++loop_idx) {
    // The order of current loop
    const auto order_idx = this->loop_order_candidates_.front()[loop_idx];

    // Leading order will not be parallelized here. In common leading case, the
    // second order in input/output tensors will not be parallelize here either.
    if ((this->in_ld_idx_ == order_idx or this->out_ld_idx_ == order_idx) or
        ((this->in_ld_idx_ + 1 == order_idx or out_2nd == order_idx) and
            this->param_->is_common_leading()))
      continue;

    // Try to parallelize current non-leading order loop
    hptc::assign_factor(this->th_factor_map_, this->avail_parallel_[order_idx],
        this->template_descriptor_.parallel_strategy[order_idx],
        hptc::ModCmp<TensorUInt>());
  }

  // Erase the zero-count factors
  auto flatted_map = hptc::flat_map(this->th_factor_map_);
  this->th_factor_map_ = hptc::factorize(
      std::accumulate(flatted_map.begin(), flatted_map.end(), 1,
          std::multiplies<TensorUInt>()));
}


template <typename ParamType>
void PlanTransOptimizer<ParamType>::init_vec_general_() {
  /*
   * Concept explanation:
   *    input leading order (continuous in input tensor)
   *  ___________________________________________________
   * |                            |           |      |   |
   * |                            |           |      |   |
   * |                            |           |      |   |
   * |                            |           |      |   |
   * |                            |           |      |   |
   * |                            |           |      |   |
   * |                            |           |      |   |
   * |             A              |     B     |      |   |
   * |                            |           |      |   |
   * |                            |           |      |   |
   * |                            |           |  E   |   |
   * |                            |           |      | H | output leading order,
   * |                            |           |      |   | (continuous in output
   * |                            |           |      |   | tensor)
   * |____________________________|___________|      |   |
   * |                            |           |      |   |
   * |                            |           |      |   |
   * |             C              |     D     |      |   |
   * |                            |           |      |   |
   * |____________________________|___________|______|   |
   * |                                        |      |   |
   * |                   F                    |  G   |   |
   * |________________________________________|______|___|
   * |                       I                           |
   * |___________________________________________________|
   *
   * The vectorization will be set as above. The vectorization order will be in
   * alphabetical order of these regions. Region description:
   * Core region: A + B + C + D, vectorized with full kernels.
   * Region A: Big core region. First choice for deploying threads.
   * Region B: Vertical core region. On the right side of big core region.
   * Region C: Horizontal core region. Under the big core region.
   * Region D: Small core region. At right bottom of the entire core region.
   *
   * Side region: E + F + G, vectorized with half kernels.
   * Region E: Vertical side region. On the right side of core regions.
   * Region F: Horizontal side region. Under the core region.
   * Region G: Small side region. At right bottom of the entire side region.
   *
   * Scalar region: H + I, not vectorized.
   * Region H: Vertical scalar region. On the right side of side region.
   * Region I: Horizontal scalar region. Under the side region.
   *
   * Naming convention in this function:
   * **_cont_**: input leading order related variables, it is continuous in
   *    memory in input tensor, but NOT continuous in output tensor
   * **_ncont_**: output leading order related variables, it is NOT continuous
   *    in memory in input tensor, but continuous in output tensor
   * **_num: kernel numbers, e.g. cont_num, number of kernels in input leading
   *    order.
   * **_len: element numbers, e.g. cont_len, number of elements in input leading
   *    order.
   * **_size: macro kernel's size, i.e. number of micro kernels tiled in a row
   *    or column, e.g. cont_kn_size = 3 means 3 micro kernels are tiled along
   *    input leading direction.
   * **_step: number of elements, it's often the loops' step length
   * knf_** and knh_**: related to full kernels and half kernels
   */

  // Prepare parameters for vectorization
  const auto knf_basic_len
      = this->param_->get_kernel().knf_basic.get_ncont_len();
  const auto knh_basic_len
      = this->param_->get_kernel().knh_basic.get_ncont_len();
  const TensorUInt knh_scale
      = this->param_->get_kernel().knh_giant.get_ncont_len() / knh_basic_len;
  const auto cont_len = this->param_->input_tensor.get_size(this->in_ld_idx_);
  const auto ncont_len = this->param_->input_tensor.get_size(this->out_ld_idx_);

  if (cont_len >= knf_basic_len and ncont_len >= knf_basic_len) {
    // Leading orders can be vectorized by full kernel
    const TensorUInt knf_scale
        = this->param_->get_kernel().knf_giant.get_ncont_len() / knf_basic_len;

    // Create rest thread number factors vector
    auto factor_map = this->th_factor_map_;

    // Assign threads factors to all loops as many as possible, large prime
    // factors may not be taken into account here.
    TensorIdx non_ld_avail = std::accumulate(this->avail_parallel_.begin(),
        this->avail_parallel_.end(), 1, std::multiplies<TensorIdx>());
    auto rest_factors = hptc::flat_map(factor_map);
    std::sort(rest_factors.begin(), rest_factors.end());
    rest_factors = hptc::approx_prod(rest_factors, non_ld_avail);

    // Get available product of non leading order loops' parallelism
    non_ld_avail /= this->avail_parallel_[this->in_ld_idx_];
    non_ld_avail /= this->avail_parallel_[this->out_ld_idx_];
    const TensorUInt knf_ncont_num = ncont_len / knf_basic_len,
        knf_cont_num = cont_len / knf_basic_len;
    std::vector<LoopParaStrategy_> loops{
        { knf_cont_num, 1, this->in_ld_idx_ },
        { knf_ncont_num, 1, this->out_ld_idx_ },
        { non_ld_avail, 1, ORDER } };

    // Assign threads to leading loops and non-leading loops
    for (auto factor : rest_factors) {
      // Sort in descending order of available parallelism, if two loops have
      // the same parallelism, put the non-leading loop or the input leading
      // loop on the left.
      std::sort(loops.begin(), loops.end(),
          [this] (const LoopParaStrategy_ &a, const LoopParaStrategy_ &b) {
              return a.size > b.size or (a.size == b.size and
              (ORDER == a.loop_idx or this->in_ld_idx_ == a.loop_idx)); });

      if (0 == loops[0].size % factor or loops[0].size / factor > loops[1].size)
        // The largest loop can be exactly divided by the factor or after force
        // dividing, it's still larger than the second largest
        loops[0].size /= factor, loops[0].th_num *= factor;
      else if (0 == loops[1].size % factor or
          loops[1].size / factor > loops[2].size)
        // The second largest loop can be exactly divided by the factor or
        // after force dividing, it's still larger than the third largest
        loops[1].size /= factor, loops[1].th_num *= factor;
      else if (0 == loops[2].size % factor)
        // The second largest loop can be exactly divided by the factor
        loops[2].size /= factor, loops[2].th_num *= factor;
      else
        // None of the above conditions is satisfied, assign to the largest loop
        loops[0].size /= factor, loops[0].th_num *= factor;
    }

    TensorUInt cont_assigned = 1, ncont_assigned = 1;
    for (auto &loop : loops) {
      if (this->in_ld_idx_ == loop.loop_idx)
        cont_assigned = loop.th_num;
      else if (this->out_ld_idx_ == loop.loop_idx)
        ncont_assigned = loop.th_num;
    }

    // Vectorization on core region
    // Split core region, small core could have zero-sizes, but big core always
    // has non-zero-sizes, because *_assigned are always <= knf_*_num
    const TensorUInt small_core_cont_num = knf_cont_num % cont_assigned,
        small_core_ncont_num = knf_ncont_num % ncont_assigned;
    const TensorUInt big_core_cont_num = knf_cont_num - small_core_cont_num,
        big_core_ncont_num = knf_ncont_num - small_core_ncont_num;

    // Calculate big core region macro kernel's size
    const auto big_core_kn_cont_size = hptc::select_kn_size(knf_scale,
        big_core_cont_num / cont_assigned);
    const auto big_core_kn_ncont_size = hptc::select_kn_size(knf_scale,
        big_core_ncont_num / ncont_assigned);

    // Update available parallelism at input and output leading order loop
    this->avail_parallel_[this->in_ld_idx_]
        = big_core_cont_num / big_core_kn_cont_size;
    this->avail_parallel_[this->out_ld_idx_]
        = big_core_ncont_num / big_core_kn_ncont_size;

    // Vectorize big core region
    this->init_vec_deploy_kernels_(KernelTypeTrans::KERNEL_FULL,
        big_core_kn_cont_size, big_core_kn_ncont_size, 0, 0, big_core_cont_num,
        big_core_ncont_num);

    // Vectorize vertical core region
    this->init_vec_deploy_kernels_(KernelTypeTrans::KERNEL_FULL, 1,
        big_core_kn_ncont_size, big_core_cont_num * knf_basic_len, 0,
        small_core_cont_num, big_core_ncont_num);

    // Vectorize horizontal core region
    this->init_vec_deploy_kernels_(KernelTypeTrans::KERNEL_FULL,
        big_core_kn_cont_size, 1, 0, big_core_ncont_num * knf_basic_len,
        big_core_cont_num, small_core_ncont_num);

    // Vectorize small core region
    this->init_vec_deploy_kernels_(KernelTypeTrans::KERNEL_FULL, 1, 1,
        big_core_cont_num * knf_basic_len, big_core_ncont_num * knf_basic_len,
        small_core_cont_num, small_core_ncont_num);

    // Vectorization on side region
    // Calculate side region macro kernel size
    const auto horiz_side_kn_cont_size = hptc::select_kn_size(knh_scale,
        knf_cont_num * 2);
    const auto vert_side_kn_ncont_size = hptc::select_kn_size(knh_scale,
        knf_ncont_num * 2);

    // Vectorize vertical side region
    this->init_vec_deploy_kernels_(KernelTypeTrans::KERNEL_HALF, 1,
        vert_side_kn_ncont_size, knf_cont_num * knf_basic_len, 0,
        (cont_len % knf_basic_len) / knh_basic_len, knf_ncont_num * 2);

    // Vectorize horizontal side region
    this->init_vec_deploy_kernels_(KernelTypeTrans::KERNEL_HALF,
        horiz_side_kn_cont_size, 1, 0, knf_ncont_num * knf_basic_len,
        knf_cont_num * 2, (ncont_len % knf_basic_len) / knh_basic_len);

    // Vectorize small side region
    this->init_vec_deploy_kernels_(KernelTypeTrans::KERNEL_HALF, 1, 1,
        knf_cont_num * knf_basic_len, knf_ncont_num * knf_basic_len,
        (cont_len % knf_basic_len) / knh_basic_len,
        (ncont_len % knf_basic_len) / knh_basic_len);

    // Set up vertical scalar region
    const TensorUInt cont_rest_len = cont_len % knh_basic_len,
        ncont_rest_len = ncont_len % knh_basic_len;
    this->init_vec_deploy_kernels_(KernelTypeTrans::KERNEL_LINE, 1, 1,
        cont_len - cont_rest_len, 0, cont_rest_len, ncont_len - ncont_rest_len);

    // Set up horizontal scalar region
    this->init_vec_deploy_kernels_(KernelTypeTrans::KERNEL_LINE, 1, 1, 0,
        ncont_len - ncont_rest_len, cont_len, ncont_rest_len, true);
  }
  else if (cont_len >= knh_basic_len and ncont_len >= knh_basic_len) {
    // Leading orders are too small for full kernels, use half kernels
    const TensorUInt knh_cont_num = cont_len / knh_basic_len,
        knh_ncont_num = ncont_len / knh_basic_len;
    const auto kn_cont_size = hptc::select_kn_size(knh_scale, knh_cont_num);
    const auto kn_ncont_size = hptc::select_kn_size(knh_scale, knh_ncont_num);

    // Update available parallelism at input and output leading order loop
    this->avail_parallel_[this->in_ld_idx_] = knh_cont_num / kn_cont_size;
    this->avail_parallel_[this->out_ld_idx_] = knh_ncont_num / kn_ncont_size;

    // Vectorize side region
    this->init_vec_deploy_kernels_(KernelTypeTrans::KERNEL_HALF, kn_cont_size,
        kn_ncont_size, 0, 0, knh_cont_num, knh_ncont_num);

    // Set up vertical scalar region
    this->init_vec_deploy_kernels_(KernelTypeTrans::KERNEL_LINE, 1, 1,
        knh_cont_num * knh_basic_len, 0, cont_len % knh_basic_len,
        knh_ncont_num * knh_basic_len);

    // Set up horizontal scalar region
    this->init_vec_deploy_kernels_(KernelTypeTrans::KERNEL_LINE, 1, 1, 0,
        knh_ncont_num * knh_basic_len, cont_len, ncont_len % knh_basic_len,
        true);
  }
  else {
    // Leading orders are too small for full kernels, use linear kernels
    this->avail_parallel_[this->in_ld_idx_] = cont_len;
    this->avail_parallel_[this->out_ld_idx_] = ncont_len;
    this->init_vec_deploy_kernels_(KernelTypeTrans::KERNEL_LINE, 1, 1, 0, 0,
        cont_len, ncont_len);
  }
}


template <typename ParamType>
void PlanTransOptimizer<ParamType>::init_vec_deploy_kernels_(
    const KernelTypeTrans kn_type, const TensorUInt kn_cont_size,
    const TensorUInt kn_ncont_size, const TensorUInt cont_begin_pos,
    const TensorUInt ncont_begin_pos, const TensorUInt cont_offset_size,
    const TensorUInt ncont_offset_size, const bool is_linh) {
  if (cont_offset_size > 0 and ncont_offset_size > 0) {
    // Locate kernel's position
    const auto kernel_offset = this->param_->get_kernel().kernel_offset(kn_type,
        kn_cont_size, kn_ncont_size, is_linh);
    auto &oper = this->template_descriptor_.description[0][kernel_offset];

    // Set up all loops
    if (oper.is_disabled()) {
      // Set up loops for leading orders
      oper.loop_begin[this->in_ld_idx_] = cont_begin_pos;
      oper.loop_step[this->in_ld_idx_]
          = this->param_->get_kernel().kn_cont_len(kn_type) * kn_cont_size;

      oper.loop_begin[this->out_ld_idx_] = ncont_begin_pos;
      oper.loop_step[this->out_ld_idx_]
          = this->param_->get_kernel().kn_ncont_len(kn_type) * kn_ncont_size;
    }
    oper.loop_end[this->in_ld_idx_] = cont_begin_pos
      + cont_offset_size * this->param_->get_kernel().kn_cont_len(kn_type);
    oper.loop_end[this->out_ld_idx_] = ncont_begin_pos
      + ncont_offset_size * this->param_->get_kernel().kn_ncont_len(kn_type);

    // Set up non leading order loops
    oper.set_pass(this->in_ld_idx_);
    for (auto loop_idx = this->in_ld_idx_ + 1; loop_idx < ORDER; ++loop_idx) {
      if (this->out_ld_idx_ == loop_idx)
        continue;

      oper.loop_begin[loop_idx] = 0;
      oper.loop_end[loop_idx] = this->param_->input_tensor.get_size(loop_idx);
      oper.loop_step[loop_idx] = 1;
    }
  }
}


template <typename ParamType>
void PlanTransOptimizer<ParamType>::init_vec_common_leading_() {
  // Prepare parameters for vectorization
  const auto cl_in_ld_idx = this->in_ld_idx_ + 1;
  const auto cl_out_ld_idx = this->in_ld_idx_
      + this->param_->perm[this->in_ld_idx_ + 1];
  const auto cl_in_ld_len = this->param_->input_tensor.get_size(cl_in_ld_idx);
  const auto cl_out_ld_len = this->param_->input_tensor.get_size(cl_out_ld_idx);
  auto factor_map = this->th_factor_map_;

  // Assign threads factors to all loops as many as possible, large prime
  // factors may not be taken into account here.
  TensorIdx non_ld_avail = std::accumulate(this->avail_parallel_.begin(),
      this->avail_parallel_.end(), 1, std::multiplies<TensorIdx>());
  auto rest_factors = hptc::flat_map(factor_map);
  std::sort(rest_factors.begin(), rest_factors.end());
  rest_factors = hptc::approx_prod(rest_factors, non_ld_avail);

  // Get available product of non leading order loops' parallelism
  non_ld_avail /= this->avail_parallel_[cl_in_ld_idx];
  non_ld_avail /= this->avail_parallel_[cl_out_ld_idx];
  std::vector<LoopParaStrategy_> loops{
      { this->avail_parallel_[cl_in_ld_idx], 1, cl_in_ld_idx },
      { this->avail_parallel_[cl_out_ld_idx], 1, cl_out_ld_idx },
      { non_ld_avail, 1, ORDER } };

  // Assign threads to leading loops and non-leading loops
  for (auto factor : rest_factors) {
    // Sort in descending order of available parallelism, if two loops have
    // the same parallelism, put the non-leading loop or the output leading
    // loop on the left
    std::sort(loops.begin(), loops.end(), [this, cl_out_ld_idx] (
          const LoopParaStrategy_ &a, const LoopParaStrategy_ &b) {
            return a.size > b.size or (a.size == b.size and
            (ORDER == a.loop_idx or cl_out_ld_idx == a.loop_idx)); });

    if (0 == loops[0].size % factor or loops[0].size / factor > loops[1].size)
      // The largest loop can be exactly divided by the factor or after force
      // dividing, it's still larger than the second largest
      loops[0].size /= factor, loops[0].th_num *= factor;
    else if (0 == loops[1].size % factor or
        loops[1].size / factor > loops[2].size)
      // The second largest loop can be exactly divided by the factor or
      // after force dividing, it's still larger than the third largest
      loops[1].size /= factor, loops[1].th_num *= factor;
    else if (0 == loops[2].size % factor)
      // The second largest loop can be exactly divided by the factor
      loops[2].size /= factor, loops[2].th_num *= factor;
    else
      // None of the above conditions is satisfied, assign to the largest loop
      loops[0].size /= factor, loops[0].th_num *= factor;
  }

  // Get the number of threads assigned to the two 2nd leading orders
  TensorUInt cl_in_ld_assigned = 1, cl_out_ld_assigned = 1;
  for (auto &loop : loops) {
    if (cl_in_ld_idx == loop.loop_idx)
      cl_in_ld_assigned = loop.th_num;
    else if (cl_out_ld_idx == loop.loop_idx)
      cl_out_ld_assigned = loop.th_num;
  }

  // Split
  const TensorUInt cl_in_ld_rest = cl_in_ld_len % cl_in_ld_assigned,
      cl_out_ld_rest = cl_out_ld_len % cl_out_ld_assigned;
  const TensorIdx cl_in_ld_num = cl_in_ld_len - cl_in_ld_rest,
      cl_out_ld_num = cl_out_ld_len - cl_out_ld_rest;
  auto cl_in_ld_size = hptc::select_kn_size(
      this->param_->get_kernel().linear_loop_max, cl_in_ld_num);
  auto cl_out_ld_size = hptc::select_kn_size(
      this->param_->get_kernel().linear_loop_max, cl_out_ld_num);

  // Update available parallelism at the two 2nd-leading orders
  this->avail_parallel_[cl_in_ld_idx] = cl_in_ld_num / cl_in_ld_size;
  this->avail_parallel_[cl_out_ld_idx] = cl_out_ld_num / cl_out_ld_size;

  // Set linear kernels
  this->param_->set_lin_wrapper_loop(cl_in_ld_size, cl_out_ld_size);

  // Set loops
  // Set core loop
  auto &core_loop = this->template_descriptor_.description[0][0];
  core_loop.set_pass(ORDER);
  for (auto loop_idx = this->in_ld_idx_ + 1; loop_idx < ORDER; ++loop_idx)
    core_loop.loop_end[loop_idx] = this->param_->input_tensor.get_size(
        loop_idx);
  core_loop.loop_end[cl_in_ld_idx] = cl_in_ld_num;
  core_loop.loop_step[cl_in_ld_idx] = cl_in_ld_size;
  core_loop.loop_end[cl_out_ld_idx] = cl_out_ld_num;
  core_loop.loop_step[cl_out_ld_idx] = cl_out_ld_size;

  // Set right side loop
  if (0 != cl_in_ld_rest) {
    auto &right_loop = this->template_descriptor_.description[0][1];
    right_loop.set_pass(ORDER);
    for (auto loop_idx = this->in_ld_idx_ + 1; loop_idx < ORDER; ++loop_idx)
      right_loop.loop_end[loop_idx] = this->param_->input_tensor.get_size(
          loop_idx);
    right_loop.loop_begin[cl_in_ld_idx] = cl_in_ld_num;
    right_loop.loop_end[cl_out_ld_idx] = cl_out_ld_num;
    right_loop.loop_step[cl_out_ld_idx] = cl_out_ld_size;
  }

  // Set bottom side loop
  if (0 != cl_out_ld_rest) {
    auto &bottom_loop = this->template_descriptor_.description[0][2];
    bottom_loop.set_pass(ORDER);
    for (auto loop_idx = this->in_ld_idx_ + 1; loop_idx < ORDER; ++loop_idx)
      bottom_loop.loop_end[loop_idx] = this->param_->input_tensor.get_size(
          loop_idx);
    bottom_loop.loop_end[cl_in_ld_idx] = cl_in_ld_num;
    bottom_loop.loop_step[cl_in_ld_idx] = cl_in_ld_size;
    bottom_loop.loop_begin[cl_out_ld_idx] = cl_out_ld_num;
  }

  // Set scalar loop
  if (0 != cl_in_ld_rest and 0 != cl_out_ld_rest) {
    auto &scalar_loop = this->template_descriptor_.description[0][3];
    scalar_loop.set_pass(ORDER);
    for (auto loop_idx = this->in_ld_idx_ + 1; loop_idx < ORDER; ++loop_idx)
      scalar_loop.loop_end[loop_idx] = this->param_->input_tensor.get_size(
          loop_idx);
    scalar_loop.loop_begin[cl_in_ld_idx] = cl_in_ld_num;
    scalar_loop.loop_begin[cl_out_ld_idx] = cl_out_ld_num;
  }
}


template <typename ParamType>
void PlanTransOptimizer<ParamType>::init_parallel_rule_general_() {
  // If there's no more thread factors or available parallelism left, then
  // return the current template version
  if (1 == this->threads_ or this->th_factor_map_.empty() or
      std::accumulate(this->avail_parallel_.begin(),
          this->avail_parallel_.end(), 1, std::multiplies<TensorUInt>()) <= 1)
    return;

  auto loop_in_ld = LoopParaStrategy_(this->avail_parallel_[this->in_ld_idx_],
      1, this->in_ld_idx_);
  auto loop_out_ld = LoopParaStrategy_(this->avail_parallel_[this->out_ld_idx_],
      1, this->out_ld_idx_);
  std::vector<LoopParaStrategy_> loop_strategies;
  for (auto loop_idx = this->in_ld_idx_ + 1; loop_idx < ORDER; ++loop_idx) {
    if (this->out_ld_idx_ == loop_idx or 1 == this->avail_parallel_[loop_idx])
      continue;
    loop_strategies.emplace_back(this->avail_parallel_[loop_idx], 1, loop_idx);
  }

  // Assign rest threads' prime factors that can be exactly divided to leadings
  const bool out_ld_large
      = this->param_->input_tensor.get_size(this->in_ld_idx_)
      <= this->param_->input_tensor.get_size(this->out_ld_idx_);
  auto &larger_loop = out_ld_large ? loop_out_ld : loop_in_ld;
  auto &smaller_loop = out_ld_large ? loop_in_ld : loop_out_ld;

  std::unordered_map<TensorUInt, TensorUInt> factor_map;
  for (auto kv : this->th_factor_map_)
    if (kv.second > 0)
      factor_map[kv.first] = kv.second;
  auto larger_assigned = hptc::assign_factor(factor_map, larger_loop.size,
      larger_loop.th_num, hptc::ModCmp<TensorUInt>());
  auto smaller_assigned = hptc::assign_factor(factor_map, smaller_loop.size,
      smaller_loop.th_num, hptc::ModCmp<TensorUInt>());

  // Sort non-leading loops by their available parallelism and order
  std::unordered_map<TensorUInt, TensorUInt> order_map;
  for (auto loop_idx = this->in_ld_idx_; loop_idx < ORDER; ++loop_idx)
    order_map[this->loop_order_candidates_.front()[loop_idx]] = loop_idx;
  std::sort(loop_strategies.begin(), loop_strategies.end(),
      [&order_map] (const LoopParaStrategy_ &a, const LoopParaStrategy_ &b) {
          return a.size > b.size or (a.size == b.size and
              order_map[a.loop_idx] < order_map[b.loop_idx]); });

  // Assign rest threads to non-leading loops
  for (auto &loop : loop_strategies)
    hptc::assign_factor(factor_map, loop.size, loop.th_num,
        std::greater_equal<TensorUInt>());

  // Give assigned leading order threads back to non-leading orders if necessary
  if (not loop_strategies.empty()) {
    // Create heap on loop parallel strategy
    auto heap_cmp =
        [&order_map] (const LoopParaStrategy_ &a, const LoopParaStrategy_ &b) {
            return a.size < b.size or (a.size == a.size and
                order_map[a.loop_idx] > order_map[b.loop_idx]); };
    std::make_heap(loop_strategies.begin(), loop_strategies.end(), heap_cmp);

    larger_loop.size *= larger_loop.th_num;
    smaller_loop.size *= smaller_loop.th_num;
    larger_loop.th_num = smaller_loop.th_num = 1;
    smaller_assigned.insert(smaller_assigned.end(), larger_assigned.begin(),
        larger_assigned.end());

    for (auto factor : smaller_assigned) {
      const bool larger_div = 0 == larger_loop.size % factor;
      const bool smaller_div = 0 == smaller_loop.size % factor;
      auto &top_loop = loop_strategies[0];
      auto &selected_loop = larger_div and smaller_div
        ? (larger_loop.size >= smaller_loop.size ? larger_loop : smaller_loop)
        : (larger_div ? larger_loop : smaller_loop);
      if (top_loop.size / factor > selected_loop.size)
        top_loop.size /= factor, top_loop.th_num *= factor;
      else
        selected_loop.size /= factor, selected_loop.th_num *= factor;
      std::make_heap(loop_strategies.begin(), loop_strategies.end(), heap_cmp);
    }
  }

  // Dealing with large prime factors
  TensorUInt rest_threads = 1, rest_avail = 1;
  for (auto kv : factor_map)
    for (TensorUInt freq = 0; freq < kv.second; ++freq)
      rest_threads *= kv.first;
  for (auto &loop : loop_strategies)
    rest_avail *= loop.size;
  rest_avail *= larger_loop.size;
  rest_avail *= smaller_loop.size;

  if (rest_threads >= rest_avail) {
    for (auto &loop : loop_strategies) {
      loop.th_num *= loop.size;
      loop.size = 1;
    }
  }
  else if (rest_threads > 1) {
    auto rest_avail_vec = hptc::flat_map(hptc::factorize(rest_avail));
    std::sort(rest_avail_vec.begin(), rest_avail_vec.end());
    auto assigned_factors = hptc::approx_prod(rest_avail_vec, rest_threads);
    factor_map.clear();
    for (auto factor : assigned_factors) {
      if (1 == factor_map.count(factor))
        ++factor_map[factor];
      else
        factor_map[factor] = 1;
    }

    for (auto &loop : loop_strategies)
      hptc::assign_factor(factor_map, loop.size, loop.th_num,
          hptc::ModCmp<TensorUInt>());
    hptc::assign_factor(factor_map, larger_loop.size, larger_loop.th_num,
        hptc::ModCmp<TensorUInt>());
    hptc::assign_factor(factor_map, smaller_loop.size, smaller_loop.th_num,
        hptc::ModCmp<TensorUInt>());
  }

  // Parallelize the default strategy
  this->template_descriptor_.parallel_strategy[this->in_ld_idx_]
      *= loop_in_ld.th_num;
  this->template_descriptor_.parallel_strategy[this->out_ld_idx_]
      *= loop_out_ld.th_num;
  for (auto &loop : loop_strategies)
    this->template_descriptor_.parallel_strategy[loop.loop_idx] *= loop.th_num;
}


template <typename ParamType>
void PlanTransOptimizer<ParamType>::init_parallel_rule_common_leading_() {
  // If there's no more thread factors or available parallelism left, then
  // return the current template version
  if (1 == this->threads_ or this->th_factor_map_.empty() or
      std::accumulate(this->avail_parallel_.begin(),
          this->avail_parallel_.end(), 1, std::multiplies<TensorUInt>()) <= 1)
    return;

  std::vector<LoopParaStrategy_> loop_strategies;
  for (auto loop_idx = this->in_ld_idx_; loop_idx < ORDER; ++loop_idx)
    loop_strategies.emplace_back(this->avail_parallel_[loop_idx], 1, loop_idx);

  // Sort loop strategies by their available parallelism
  std::unordered_map<TensorUInt, TensorUInt> order_map;
  for (auto loop_idx = this->in_ld_idx_; loop_idx < ORDER; ++loop_idx)
    order_map[this->loop_order_candidates_.front()[loop_idx]] = loop_idx;
  std::sort(loop_strategies.begin(), loop_strategies.end(),
      [&order_map] (const LoopParaStrategy_ &a, const LoopParaStrategy_ &b) {
          return a.size > b.size or (a.size == b.size and
              order_map[a.loop_idx] < order_map[b.loop_idx]); });

  // Assign rest threads and compute rest available parallelism
  auto factor_map = this->th_factor_map_;
  TensorUInt rest_avail = 1;
  for (auto &loop : loop_strategies) {
    hptc::assign_factor(factor_map, loop.size, loop.th_num,
        std::greater_equal<TensorUInt>());
    rest_avail *= loop.size;
  }

  // Dealing with large prime factors
  TensorUInt rest_threads = 1;
  for (auto kv : factor_map)
    for (TensorUInt freq = 0; freq < kv.second; ++freq)
      rest_threads *= kv.first;

  if (rest_threads >= rest_avail) {
    for (auto &loop : loop_strategies) {
      loop.th_num *= loop.size;
      loop.size = 1;
    }
  }
  else if (rest_threads > 1) {
    //auto factorized_avail = hptc::factorize(rest_avail);
    auto rest_avail_vec = hptc::flat_map(hptc::factorize(rest_avail));
    std::sort(rest_avail_vec.begin(), rest_avail_vec.end());
    auto assigned_factors = hptc::approx_prod(rest_avail_vec, rest_threads);
    factor_map.clear();
    for (auto factor : assigned_factors) {
      if (1 == factor_map.count(factor))
        ++factor_map[factor];
      else
        factor_map[factor] = 1;
    }

    for (auto &loop : loop_strategies)
      hptc::assign_factor(factor_map, loop.size, loop.th_num,
          hptc::ModCmp<TensorUInt>());
  }

  // Parallelize the default strategy
  for (auto &loop : loop_strategies)
    this->template_descriptor_.parallel_strategy[loop.loop_idx] *= loop.th_num;
}


template <typename ParamType>
void PlanTransOptimizer<ParamType>::init_parallel_heur_(
    const TensorInt tune_num, const TensorInt heur_num) {
  // If there's no more thread factors or available parallelism left, then
  // return the current template version
  if (1 == this->threads_ or this->th_factor_map_.empty() or
      std::accumulate(this->avail_parallel_.begin(),
          this->avail_parallel_.end(), 1, std::multiplies<TensorUInt>()) <= 1)
    return;

  // Data structure for describing available parallelism at a loop
  struct LoopBin {
    LoopBin(TensorUInt order_idx, TensorUInt capacity) : order_idx(order_idx),
        capacity(capacity), curr_size(1) {}
    const TensorUInt order_idx;
    const TensorIdx capacity;
    TensorUInt curr_size;
  };

  std::vector<LoopBin> loop_bins;
  for (auto order_idx = this->in_ld_idx_; order_idx < ORDER; ++order_idx)
    if (this->avail_parallel_[order_idx] > 1)
      loop_bins.emplace_back(order_idx, this->avail_parallel_[order_idx]);
  const auto bin_num = static_cast<TensorInt>(loop_bins.size());

  // Data structure for describing a thread factor's position
  struct FactorPos {
    FactorPos(TensorUInt factor) : factor(factor), bin_idx(-1) {}
    const TensorUInt factor;
    TensorInt bin_idx;
  };

  /*
   * Set up two stacks for recording factors' status, deployment stack records
   * the factors that assigned to a certain loop, wait stack records the factors
   * that not assigned to any loops.
   */
  std::stack<FactorPos> deploy_stack, wait_stack;
  auto thread_factors = hptc::flat_map(this->th_factor_map_);

  // Sort thread factors in ascending order before pushing them in stack
  std::sort(thread_factors.begin(), thread_factors.end(),
      std::greater<TensorUInt>());
  for (auto factor : thread_factors)
    wait_stack.emplace(factor);
  deploy_stack.push(wait_stack.top());
  wait_stack.pop();

  // Create heap to preserve the best $tune_num resutls
  // "Cost-Parallel" pair: (cost, parallel strategy)
  using ParaDes = std::pair<double, ParaStrategyTrans<ORDER>>;
  auto heap_cmp = [] (const ParaDes &a, const ParaDes &b) -> bool {
      return a.first < b.first; };
  std::priority_queue<ParaDes, std::vector<ParaDes>, decltype(heap_cmp)>
      best_heap(heap_cmp);

  TensorInt times = 0;
  while (not deploy_stack.empty() and (heur_num < 0 or heur_num > times)) {
    auto &bin_idx = deploy_stack.top().bin_idx;
    // Find current assigned factor's next position
    if (-1 == bin_idx)
      ++bin_idx;
    else if (bin_idx >= bin_num)
      bin_idx -= bin_num;
    else {
      loop_bins[bin_idx].curr_size /= deploy_stack.top().factor;
      ++bin_idx;
    }

    for (; bin_idx < bin_num and
        loop_bins[bin_idx].curr_size * deploy_stack.top().factor
        > loop_bins[bin_idx].capacity; ++bin_idx);

    if (bin_idx == bin_num) {
      // No next suitable position, pop back to wait stack
      bin_idx = -1;
      wait_stack.push(deploy_stack.top());
      deploy_stack.pop();
    }
    else {
      // Find next suitable position, deploy this factor
      loop_bins[bin_idx].curr_size *= deploy_stack.top().factor;
      if (not wait_stack.empty()) {
        // The wait stack is NOT empty, push its top factor into deploy stack
        // If wait stack's top has same value with deploy stack, set wait stack
        // top's bin_idx with -2;
        if (wait_stack.top().factor == deploy_stack.top().factor)
          wait_stack.top().bin_idx = deploy_stack.top().bin_idx + bin_num;
        deploy_stack.push(wait_stack.top());
        wait_stack.pop();
      }
      else {
        // The wait stack is empty, we got a parallelization strategy
        ParaStrategyTrans<ORDER> strategy;
        strategy.fill(1);
        for (auto &bin : loop_bins) {
          strategy[bin.order_idx] *= bin.curr_size;
        }

        // Calculate cost and push into best heap
        auto new_cost = this->heur_parallel_evaluator_(strategy);
        if (tune_num < 0 or tune_num > static_cast<TensorInt>(best_heap.size()))
          best_heap.emplace(new_cost, strategy);
        else if (best_heap.top().first > new_cost) {
          best_heap.pop();
          best_heap.emplace(new_cost, strategy);
        }

        // Update times
        ++times;
      }
    }
  }

  // If the best heap is empty, it means that there could be large prime in
  // thread factors. In this case, use the default parallelization strategy.
  // Store result
  while (not best_heap.empty()) {
    this->parallel_strategy_candidates_.emplace_back(best_heap.top().second);
    best_heap.pop();

    // Multiply with template descriptor's parallelization strategy
    for (TensorUInt order_idx = 0; order_idx < ORDER; ++order_idx)
      this->parallel_strategy_candidates_.back()[order_idx]
          *= this->template_descriptor_.parallel_strategy[order_idx];
  }
}


template <typename ParamType>
double PlanTransOptimizer<ParamType>::heur_loop_evaluator_(
    const LoopOrderTrans<ORDER> &target_loop_order) const {
  // Locate begin index
  const auto merged_order = this->param_->merged_order;

  // Create loop penalty array
  // [..., 2 * penalty_step, 1 * penalty_step, 0 * penalty_step]
  std::vector<double> loop_penalty(merged_order, 0.0);
  loop_penalty.back() = this->heur_loop_penalty_begin;
  for (TensorInt loop_idx = merged_order - 2; loop_idx >= 0; --loop_idx)
    loop_penalty[loop_idx] = loop_penalty[loop_idx + 1]
        + this->heur_loop_penalty_step;

  // Create target loop order's index map
  // Key is a specific tensor order, values is its level in loop
  std::vector<TensorUInt> target_map(merged_order, 0);
  for (auto loop_idx = this->in_ld_idx_; loop_idx < ORDER; ++loop_idx)
    target_map[target_loop_order[loop_idx] - this->in_ld_idx_]
        = loop_idx - this->in_ld_idx_;

  // Compute target loop order's costs
  double loop_cost = 0.0, importance = this->heur_loop_importance_begin;
  for (TensorUInt order_idx = 0; order_idx < merged_order;
      ++order_idx, importance *= this->heur_loop_importance_scale) {
    auto input_order_loop_pos = target_map[order_idx];
    auto abs_idx = order_idx + this->in_ld_idx_;
    auto output_order_loop_pos = target_map[this->param_->perm[abs_idx]];

    auto input_order_penalty = loop_penalty[input_order_loop_pos];
    auto output_order_penalty = loop_penalty[output_order_loop_pos];
    loop_cost += importance
        * (input_order_penalty * this->heur_loop_input_penalty_factor
            + output_order_penalty * this->heur_loop_output_penalty_factor);
  }

  return loop_cost;
}


template <typename ParamType>
double PlanTransOptimizer<ParamType>::heur_parallel_evaluator_(
    const ParaStrategyTrans<ORDER> &target_para) const {
  // Calculate costs
  auto cost = this->heur_para_cost_begin;
  for (auto loop_idx = this->in_ld_idx_; loop_idx < ORDER; ++loop_idx) {
    if (target_para[loop_idx] <= 1)
      continue;

    // Calculate number of steps assigned to each thread
    TensorIdx steps_per_thread
        = (this->avail_parallel_[loop_idx] + target_para[loop_idx] - 1)
            / target_para[loop_idx];
    // Calculate load for loop at level loop_idx given certain number of threads
    const auto load = steps_per_thread * target_para[loop_idx];
    // Update cost
    cost *= static_cast<double>(load) / this->avail_parallel_[loop_idx];
  }

  // Strongly penalize parallelization at stride-1 loop in common leading case
  if (this->param_->is_common_leading())
    cost *= std::pow(this->heur_para_penalty_factor_cl,
        target_para[this->in_ld_idx_] - 1);

  // Penalize parallelization at input/output leading order loop
  cost *= std::pow(this->heur_para_penalty_factor_inld,
      std::min(this->heur_para_max_penalty_threads,
          target_para[this->in_ld_idx_] - 1));
  cost *= std::pow(this->heur_para_penalty_factor_outld,
      std::min(this->heur_para_max_penalty_threads,
          target_para[this->param_->perm[this->in_ld_idx_]] - 1));

  return cost;
}


template <typename ParamType>
std::vector<typename CGraphTrans<ParamType>::Descriptor>
PlanTransOptimizer<ParamType>::gen_candidates_() const {
  std::vector<typename CGraphTrans<ParamType>::Descriptor> candidates;

  // Permute over different loop orders and parallelization strategies
  for (const auto &loop : this->loop_order_candidates_) {
    for (const auto &strategy : this->parallel_strategy_candidates_) {
      // Create a new candidate from default single threaded descriptor
      candidates.emplace_back(this->template_descriptor_);
      candidates.back().loop_order = loop;
      candidates.back().parallel_strategy = strategy;

      // Parallelization
      auto &des = candidates.back().description;

      // Calculate actual thread number and resize description
      const TensorUInt threads = std::accumulate(strategy.begin(),
          strategy.end(), 1, std::multiplies<TensorUInt>());
      des.resize(threads, des[0]);

      // Parallelize
      for (TensorUInt kn_idx = 0, kn_num = des[0].size(); kn_idx < kn_num;
          ++kn_idx) {
        // Skip disabled kernel
        auto &kn_template = des[0][kn_idx];
        if (kn_template.is_disabled())
          continue;

        for (TensorUInt loop_idx = this->in_ld_idx_, left_threads = threads;
            loop_idx < ORDER and left_threads > 1; ++loop_idx) {
          // Compute step times at current loop level
          TensorIdx steps = (kn_template.loop_end[loop_idx]
              - kn_template.loop_begin[loop_idx])
              / kn_template.loop_step[loop_idx];

          // Create vector to store steps for each thread at current loop level
          const auto curr_para = strategy[loop_idx];
          std::vector<TensorIdx> split_steps(curr_para,
              curr_para <= steps ? steps / curr_para : 0);
          std::for_each(split_steps.end() - steps % curr_para,
              split_steps.end(), [] (TensorIdx &num) { ++num; });

          // Create unit spans
          std::vector<TensorIdx> unit_begins(left_threads),
              unit_ends(left_threads);
          for (TensorIdx cp_idx = 0,
              begin_val = kn_template.loop_begin[loop_idx],
              copies = left_threads / curr_para; cp_idx < curr_para; ++cp_idx) {
            auto cp_beg = cp_idx * copies;
            auto cp_end = cp_beg + copies;
            auto end_val = begin_val
                + split_steps[cp_idx] * kn_template.loop_step[loop_idx];
            std::fill(unit_begins.begin() + cp_beg,
                unit_begins.begin() + cp_end, begin_val);
            std::fill(unit_ends.begin() + cp_beg, unit_ends.begin() + cp_end,
                end_val);
            begin_val = end_val;
          }

          // Assign rest index in loop to threads
          std::vector<TensorIdx> begins(threads), ends(threads);
          for (TensorUInt off = 0; off < threads; off += left_threads) {
            std::copy(unit_begins.begin(), unit_begins.end(),
                begins.begin() + off);
            std::copy(unit_ends.begin(), unit_ends.end(), ends.begin() + off);
          }

          for (TensorUInt th_idx = 0; th_idx < threads; ++th_idx) {
            des[th_idx][kn_idx].loop_begin[loop_idx] = begins[th_idx];
            des[th_idx][kn_idx].loop_end[loop_idx] = ends[th_idx];
            des[th_idx][kn_idx].loop_step[loop_idx]
                = kn_template.loop_step[loop_idx];
          }

          // Update left thread number
          left_threads /= curr_para;
        }
      }
    }
  }

  return candidates;
}


/*
 * Import explicit instantiation declaration for class PlanTransOptimizer, this
 * file should be generated by cmake script.
 */
#include <hptc/gen/plan_trans_util_gen.tcc>

#endif // HPTC_PLAN_PLAN_TRANS_UTIL_TCC_
