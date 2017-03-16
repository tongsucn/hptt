#pragma once
#ifndef HPTC_PLAN_PLAN_TRANS_UTIL_TCC_
#define HPTC_PLAN_PLAN_TRANS_UTIL_TCC_

template <typename ParamType>
PlanTransOptimizer<ParamType>::PlanTransOptimizer(
    const std::shared_ptr<ParamType> &param, GenNumType thread_num)
    : param_(param->merged_order <= 1 ? nullptr : param),
      threads_(thread_num),
      descriptor_(),
      th_fact_map_(),
      avail_parallel_(),
      parallel_template_() {
  if (nullptr == this->param_)
    return;

  this->init_();
}


template <typename ParamType>
std::vector<Descriptor<ParamType>>
PlanTransOptimizer<ParamType>::get_optimal(TensorIdx heur_loop_num,
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


template <typename ParamType>
void PlanTransOptimizer<ParamType>::init_() {
  // Initialize all kinds of parameters and configurations
  this->init_config_();

  // Initialize thread number
  this->init_threads_();

  // Check input and output leading and initialize vectorization
  if (this->param_->is_common_leading())
    // Input and output tensor's leading order ARE the same.
    this->init_vec_common_leading_();
  else
    // Input and output tensor's leading order ARE NOT the same.
    this->init_vec_();
/*
  std::cout << "Descriptor:" << std::endl;
  for (auto d : this->descriptor_.description[0]) {
    if (d.is_disabled())
      continue;
    std::cout << "====" << std::endl;
    std::cout << "  Begin: ";
    for (auto i = 0; i < ORDER; ++i)
      std::cout << d.loop_begin[i] << " ";
    std::cout << std::endl;
    std::cout << "  End: ";
    for (auto i = 0; i < ORDER; ++i)
      std::cout << d.loop_end[i] << " ";
    std::cout << std::endl;
    std::cout << "  Step: ";
    for (auto i = 0; i < ORDER; ++i)
      std::cout << d.loop_step[i] << " ";
    std::cout << std::endl;
  }*/

  // Initialize default loop order
  this->init_loop_();

  // Initialize parallelization (expand descriptor)
  this->init_parallel_();
}


template <typename ParamType>
void PlanTransOptimizer<ParamType>::init_config_() {
  // Initialize loop evaluator's parameters
  this->init_loop_evaluator_param_();

  // Initialize parallelization evaluator's parameters
  this->init_parallel_evaluator_param_();
}


template <typename ParamType>
void PlanTransOptimizer<ParamType>::init_loop_evaluator_param_() {
  this->penalty_begin = 0.0, this->penalty_step = 20.0;
  this->importance_begin = 1.0, this->importance_scale = 0.5;
  this->input_penalty_factor = 1.0, this->output_penalty_factor = 1.01;
  this->in_ld_award = 0.8, this->out_ld_award = 0.85;
}


template <typename ParamType>
void PlanTransOptimizer<ParamType>::init_parallel_evaluator_param_() {
  this->penalty_factor_cl = 1.01;
  this->penalty_factor_inld = 1.00010, this->penalty_factor_outld = 1.00015;
  this->max_penalty_threads = 16;
}


template <typename ParamType>
void PlanTransOptimizer<ParamType>::init_threads_() {
  // If the input thread number is zero, set thread number to maximum
  if (0 == this->threads_)
    this->threads_ = omp_get_max_threads();
  // If OpenMP returns bad number, set to single thread
  if (this->threads_ <= 0)
    this->threads_ = 1;

  // Integer factorization of thread number
  this->th_fact_map_ = hptc::factorize(this->threads_);
  this->avail_parallel_.fill(1);
  this->parallel_template_.fill(1);

  // Parallel non-leading order loops that can be exactly divided, and
  // initialize available parallelism at every loop
  const auto in_ld_idx = this->param_->begin_order_idx;
  const auto out_ld_idx = this->param_->perm[in_ld_idx] + in_ld_idx;
  std::unordered_set<GenNumType> drain_factors;
  for (auto loop_idx = in_ld_idx + 1; loop_idx < ORDER; ++loop_idx) {
    // Set available parallelism at loop level loop_idx
    this->avail_parallel_[loop_idx]
        = this->param_->input_tensor.get_size()[loop_idx];
    // Skip leading orders
    if (out_ld_idx == loop_idx)
      continue;

    // Try to parallelize current non-leading order loop
    hptc::assign_factor(drain_factors, this->th_fact_map_,
        this->avail_parallel_[loop_idx],
        this->parallel_template_[loop_idx],
        [] (const auto a, const auto b) -> bool { return 0 == a % b; });
  }

  // Remove drained factors from map
  for (auto factor : drain_factors)
    this->th_fact_map_.erase(factor);
}


template <typename ParamType>
void PlanTransOptimizer<ParamType>::init_vec_() {
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
   * **_len: element numbers, e.g. cont_len, number of elements in input leading
   *    order.
   * **_size: macro kernel's size, i.e. number of micro kernels tiled in a row
   *    or column, e.g. cont_kn_size = 3 means 3 micro kernels are tiled along
   *    input leading direction.
   * **_step: number of elements, it's often the loops' step length
   * knf_** and knh_**: related to full kernels and half kernels
   */

  // Prepare parameters for vectorization
  const auto in_ld_idx = this->param_->begin_order_idx;
  const auto out_ld_idx = this->param_->perm[in_ld_idx] + in_ld_idx;
  const auto knf_basic_len = this->param_->kn.knf_basic.get_ncont_len();
  const auto knh_basic_len = this->param_->kn.knh_basic.get_ncont_len();
  const GenNumType knh_scale
      = this->param_->kn.knh_giant.get_ncont_len() / knh_basic_len;
  const auto cont_len = this->param_->get_leading().first;
  const auto ncont_len = this->param_->get_leading().second;

  // Lambda for calculating kernel size
  auto kn_size = [] (GenNumType size, const GenNumType chunk_size) {
    while (size > 1 and 0 != chunk_size % size)
      --size;
    return size;
  };

  if (cont_len >= knf_basic_len and ncont_len >= knf_basic_len) {
    // Leading orders can be vectorized by full kernel
    const GenNumType knf_scale
        = this->param_->kn.knf_giant.get_ncont_len() / knf_basic_len;

    // Create rest thread number factors vector
    std::vector<GenNumType> rest_factors;
    for (auto kv : this->th_fact_map_)
      for (auto times = 0; times < kv.second; ++times)
        rest_factors.push_back(kv.first);
    auto fact_map = this->th_fact_map_;

    // Lambda for calculating core region vectorization
    auto vec_core = [&rest_factors, &fact_map] (const GenNumType kn_size) {
      // Get thread number prime factors that can be assigned on output leading
      std::sort(rest_factors.begin(), rest_factors.end());
      auto threads = hptc::approx_prod(rest_factors, kn_size);
      auto assigned = std::accumulate(threads.begin(), threads.end(), 1,
          std::multiplies<GenNumType>());

      if (assigned > 1) {
        // Assigned more than one threads on output leading order
        std::vector<GenNumType> drain_factors;
        for (auto factor : threads) {
          --fact_map[factor];
          if (0 == fact_map[factor])
            drain_factors.push_back(factor);
        }
        for (auto factor : drain_factors)
          fact_map.erase(factor);
        rest_factors.clear();
        for (auto kv : fact_map)
          for (auto times = 0; times < kv.second; ++times)
            rest_factors.push_back(kv.first);
      }

      return assigned;
    };

    // Vectorize the larger leading order first
    const GenNumType knf_ncont_size = ncont_len / knf_basic_len,
        knf_cont_size = cont_len / knf_basic_len;
    GenNumType ncont_assigned, cont_assigned;
    if (knf_ncont_size >= knf_cont_size) {
      ncont_assigned = vec_core(knf_ncont_size);
      cont_assigned = vec_core(knf_cont_size);
    }
    else {
      cont_assigned = vec_core(knf_cont_size);
      ncont_assigned = vec_core(knf_ncont_size);
    }

    // Vectorization on core region
    // Split core region, small core could have zero-sizes, but big core always
    // has non-zero-sizes, because *_assigned are always <= knf_*_size
    const GenNumType small_core_cont_size = knf_cont_size % cont_assigned,
        small_core_ncont_size = knf_ncont_size % ncont_assigned;
    const GenNumType big_core_cont_size = knf_cont_size - small_core_cont_size,
        big_core_ncont_size = knf_ncont_size - small_core_ncont_size;

    // Calculate big core region macro kernel's size
    const auto big_core_kn_cont_size = kn_size(knf_scale,
        big_core_cont_size / cont_assigned);
    const auto big_core_kn_ncont_size = kn_size(knf_scale,
        big_core_ncont_size / ncont_assigned);

    // Update available parallelism at input and output leading order loop
    this->avail_parallel_[in_ld_idx]
        = big_core_cont_size / big_core_kn_cont_size;
    this->avail_parallel_[out_ld_idx]
        = big_core_ncont_size / big_core_kn_ncont_size;

    // Vectorize big core region
    this->init_vec_deploy_kernels_(KernelTypeTrans::KERNEL_FULL,
        big_core_kn_cont_size, big_core_kn_ncont_size, 0, 0, big_core_cont_size,
        big_core_ncont_size);

    // Vectorize vertical core region
    this->init_vec_deploy_kernels_(KernelTypeTrans::KERNEL_FULL, 1,
        big_core_kn_ncont_size, big_core_cont_size * knf_basic_len, 0,
        small_core_cont_size, big_core_ncont_size);

    // Vectorize horizontal core region
    this->init_vec_deploy_kernels_(KernelTypeTrans::KERNEL_FULL,
        big_core_kn_cont_size, 1, 0, big_core_ncont_size * knf_basic_len,
        big_core_cont_size, small_core_ncont_size);

    // Vectorize small core region
    this->init_vec_deploy_kernels_(KernelTypeTrans::KERNEL_FULL, 1, 1,
        big_core_cont_size * knf_basic_len, big_core_ncont_size * knf_basic_len,
        small_core_cont_size, small_core_ncont_size);

    // Vectorization on side region
    // Calculate side region macro kernel size
    const auto horiz_side_kn_cont_size = kn_size(knh_scale, knf_cont_size * 2);
    const auto vert_side_kn_ncont_size = kn_size(knh_scale, knf_ncont_size * 2);

    // Vectorize vertical side region
    this->init_vec_deploy_kernels_(KernelTypeTrans::KERNEL_HALF, 1,
        vert_side_kn_ncont_size, knf_cont_size * knf_basic_len, 0,
        (cont_len % knf_basic_len) / knh_basic_len, knf_ncont_size * 2);

    // Vectorize horizontal side region
    this->init_vec_deploy_kernels_(KernelTypeTrans::KERNEL_HALF,
        horiz_side_kn_cont_size, 1, 0, knf_ncont_size * knf_basic_len,
        knf_cont_size * 2, (ncont_len % knf_basic_len) / knh_basic_len);

    // Vectorize small side region
    this->init_vec_deploy_kernels_(KernelTypeTrans::KERNEL_HALF, 1, 1,
        knf_cont_size * knf_basic_len, knf_ncont_size * knf_basic_len,
        (cont_len % knf_basic_len) / knh_basic_len,
        (ncont_len % knf_basic_len) / knh_basic_len);

    // Set up vertical scalar region
    const GenNumType cont_rest_len = cont_len % knh_basic_len,
        ncont_rest_len = ncont_len % knh_basic_len;
    this->init_vec_deploy_kernels_(KernelTypeTrans::KERNEL_LINE, 1, 1,
        cont_len - cont_rest_len, 0, cont_rest_len, ncont_len - ncont_rest_len);

    // Set up horizontal scalar region
    this->init_vec_deploy_kernels_(KernelTypeTrans::KERNEL_LINE, 1, 1, 0,
        ncont_len - ncont_rest_len, cont_len, ncont_rest_len, true);
  }
  else if (cont_len >= knh_basic_len and ncont_len >= knh_basic_len) {
    // Leading orders are too small for full kernels, use half kernels
    const GenNumType knh_cont_size = cont_len / knh_basic_len,
        knh_ncont_size = ncont_len / knh_basic_len;
    const auto kn_cont_size = kn_size(knh_scale, knh_cont_size);
    const auto kn_ncont_size = kn_size(knh_scale, knh_ncont_size);

    // Update available parallelism at input and output leading order loop
    this->avail_parallel_[in_ld_idx] = knh_cont_size / kn_cont_size;
    this->avail_parallel_[out_ld_idx] = knh_ncont_size / kn_ncont_size;

    // Vectorize side region
    this->init_vec_deploy_kernels_(KernelTypeTrans::KERNEL_HALF, kn_cont_size,
        kn_ncont_size, 0, 0, knh_cont_size, knh_ncont_size);

    // Set up vertical scalar region
    this->init_vec_deploy_kernels_(KernelTypeTrans::KERNEL_LINE, 1, 1,
        knh_cont_size * knh_basic_len, 0, cont_len % knh_basic_len,
        knh_ncont_size * knh_basic_len);

    // Set up horizontal scalar region
    this->init_vec_deploy_kernels_(KernelTypeTrans::KERNEL_LINE, 1, 1, 0,
        knh_ncont_size * knh_basic_len, cont_len, ncont_len % knh_basic_len,
        true);
  }
  else {
    // Leading orders are too small for full kernels, use linear kernels
    this->avail_parallel_[in_ld_idx] = cont_len;
    this->avail_parallel_[out_ld_idx] = ncont_len;
    this->init_vec_deploy_kernels_(KernelTypeTrans::KERNEL_LINE, 1, 1, 0, 0,
        cont_len, ncont_len);
  }
}


template <typename ParamType>
void PlanTransOptimizer<ParamType>::init_vec_deploy_kernels_(
    const KernelTypeTrans kn_type, const GenNumType kn_cont_size,
    const GenNumType kn_ncont_size, const TensorOrder cont_begin_pos,
    const TensorOrder ncont_begin_pos, const TensorOrder cont_offset_size,
    const TensorOrder ncont_offset_size, const bool is_linh) {
  /*
   * Naming convention follows the rule described in function init_vec_
   */
  if (cont_offset_size > 0 and ncont_offset_size > 0) {
    // Locate kernel's position
    const auto cont_loop_idx = this->param_->begin_order_idx;
    const auto ncont_loop_idx = this->param_->perm[cont_loop_idx]
        + cont_loop_idx;
    const auto kernel_offset = this->param_->kn.kernel_offset(kn_type,
        kn_cont_size, kn_ncont_size, is_linh);
    auto &oper = this->descriptor_.description[0][kernel_offset];

    // Set up all loops
    if (oper.is_disabled()) {
      // Set up loops for leading orders
      oper.loop_begin[cont_loop_idx] = cont_begin_pos;
      oper.loop_step[cont_loop_idx] = this->param_->kn.kn_cont_len(kn_type,
          kn_cont_size);

      oper.loop_begin[ncont_loop_idx] = ncont_begin_pos;
      oper.loop_step[ncont_loop_idx] = this->param_->kn.kn_ncont_len(kn_type,
          kn_ncont_size);
    }
    oper.loop_end[cont_loop_idx] = cont_begin_pos
      + cont_offset_size * this->param_->kn.kn_cont_len(kn_type, 1);
    oper.loop_end[ncont_loop_idx] = ncont_begin_pos
      + ncont_offset_size * this->param_->kn.kn_ncont_len(kn_type, 1);

    // Set up non leading order loops
    oper.set_pass(cont_loop_idx);
    for (auto loop_idx = cont_loop_idx + 1; loop_idx < ORDER; ++loop_idx) {
      if (ncont_loop_idx == loop_idx)
        continue;

      oper.loop_begin[loop_idx] = 0;
      oper.loop_end[loop_idx] = this->param_->input_tensor.get_size()[loop_idx];
      oper.loop_step[loop_idx] = 1;
    }
  }
}


template <typename ParamType>
void PlanTransOptimizer<ParamType>::init_vec_common_leading_() {
  auto &loop = this->descriptor_.description[0][0];
  // Set loops for non-leading orders
  loop.set_pass(ORDER);
  this->avail_parallel_[this->param_->begin_order_idx] = 1;
  for (auto loop_idx = this->param_->begin_order_idx + 1; loop_idx < ORDER;
      ++loop_idx)
    loop.loop_end[loop_idx] = this->param_->input_tensor.get_size()[loop_idx];
}


template <typename ParamType>
void PlanTransOptimizer<ParamType>::init_loop_() {
  // Data structure for describing loop, first is the order a loop stands for
  // second is loop's score
  using LoopScore = std::pair<TensorOrder, double>;

  // Locate loop re-order position (first loop that need to be re-ordered)
  const auto in_ld_idx = this->param_->begin_order_idx;
  const auto out_ld_idx = this->param_->perm[in_ld_idx] + in_ld_idx;

  // Create and initialize loop description array
  std::array<LoopScore, ORDER> scores;
  scores.fill(LoopScore(0, 0.0));
  for (TensorOrder loop_idx = in_ld_idx; loop_idx < ORDER; ++loop_idx) {
    auto curr_score = static_cast<double>(loop_idx - in_ld_idx);
    scores[loop_idx].first = loop_idx;
    scores[loop_idx].second += curr_score;
    auto perm_idx = this->param_->perm[loop_idx] + in_ld_idx;
    scores[perm_idx].second += curr_score;
  }
  scores[in_ld_idx].second *= this->in_ld_award;
  scores[out_ld_idx].second *= this->out_ld_award;

  // Sort orders according to the scores, if two orders have same score, then
  // they will be sorted by their available parallelism
  std::sort(scores.begin() + in_ld_idx, scores.end(),
      [this] (auto &a, auto &b) { return a.second > b.second or
        (a.second == b.second and
          this->avail_parallel_[a.first] < this->avail_parallel_[b.first]); });

  // Set loop order
  for (TensorOrder loop_idx = 0; loop_idx < ORDER; ++loop_idx)
    this->descriptor_.loop_order[loop_idx] = scores[loop_idx].first;
}


template <typename ParamType>
void PlanTransOptimizer<ParamType>::init_parallel_() {
  // Check rest thread resource, return if no threads left
  if (1 == this->threads_ or this->th_fact_map_.empty()) {
    std::copy(this->parallel_template_.begin(), this->parallel_template_.end(),
        this->descriptor_.parallel_strategy.begin());
    return;
  }

  /*
   * Create data structure storing parallelization information. It contains
   * three members. The size ((end - begin) / step) of a loop, number of threads
   * assigned to this loop, the original index of this loop.
   */
  struct LoopParaStategy {
    LoopParaStategy(TensorOrder size, GenNumType th_num, TensorOrder loop_idx)
        : size(size), th_num(th_num), loop_idx(loop_idx) {}
    TensorOrder size;
    GenNumType th_num;
    TensorOrder loop_idx;
  };

  const auto in_ld_idx = this->param_->begin_order_idx;
  const auto out_ld_idx = this->param_->perm[in_ld_idx] + in_ld_idx;
  std::vector<LoopParaStategy> loop_nld;
  for (TensorOrder loop_idx = 0; loop_idx < ORDER; ++loop_idx) {
    if (in_ld_idx == loop_idx or out_ld_idx == loop_idx or
        1 == this->avail_parallel_[loop_idx])
      continue;
    loop_nld.emplace_back(this->avail_parallel_[loop_idx], 1, loop_idx);
  }
  LoopParaStategy loop_in_ld = {
          this->avail_parallel_[in_ld_idx], 1, in_ld_idx },
      loop_out_ld = {
          this->avail_parallel_[out_ld_idx], 1, out_ld_idx };

  // Assign rest threads' prime factors to leading orders
  const bool out_ld_large
      = this->param_->get_leading().first <= this->param_->get_leading().second;
  auto &larger_loop = out_ld_large ? loop_out_ld : loop_in_ld;
  auto &smaller_loop = this->param_->is_common_leading()
      ? larger_loop : (out_ld_large ? loop_in_ld : loop_out_ld);

  auto fact_map = this->th_fact_map_;
  std::unordered_set<GenNumType> drain_factors;
  auto cmp_mod = [] (const auto a, const auto b) -> bool { return 0 == a % b; };
  auto larger_assigned = hptc::assign_factor(drain_factors, fact_map,
      larger_loop.size, larger_loop.th_num, cmp_mod);
  auto smaller_assigned = hptc::assign_factor(drain_factors, fact_map,
      smaller_loop.size, smaller_loop.th_num, cmp_mod);
  for (auto drained : drain_factors)
    fact_map.erase(drained);
  drain_factors.clear();

  // Sort non-leading loops by their available parallelism and order
  std::sort(loop_nld.begin(), loop_nld.end(),
      [] (const auto &a, const auto &b) -> bool { return a.size > b.size or
          (a.size == a.size and a.loop_idx > b.loop_idx); });

  // Assign rest threads to non-leading loops
  for (auto &loop : loop_nld)
    hptc::assign_factor(drain_factors, fact_map, loop.size, loop.th_num,
        std::greater<GenNumType>());
  for (auto drained : drain_factors)
    fact_map.erase(drained);
  drain_factors.clear();

  // Rests in the map should be large thread prime factors
  std::vector<GenNumType> large_primes;
  for (auto &fact : fact_map)
    for (; fact.second > 0; --fact.second)
      large_primes.push_back(fact.first);
  fact_map.clear();

  // Sort the non-leading loops again
  std::sort(loop_nld.begin(), loop_nld.end(),
      [] (const auto &a, const auto &b) -> bool { return a.size > b.size or
          (a.size == a.size and a.loop_idx > b.loop_idx); });

  // Assign assigned leading order threads to non-leading order loops
  if (not this->param_->is_common_leading()) {
    larger_assigned.reserve(larger_assigned.size() + smaller_assigned.size());
    larger_assigned.insert(larger_assigned.end(), smaller_assigned.begin(),
        smaller_assigned.end());
    smaller_loop.size *= smaller_loop.th_num, smaller_loop.th_num = 1;
  }
  larger_loop.size *= larger_loop.th_num, larger_loop.th_num = 1;

  for (auto factor : larger_assigned) {
    if (factor <= 1)
      continue;
    if (1 == fact_map.count(factor))
      ++fact_map[factor];
    else
      fact_map[factor] = 1;
  }
  for (auto &loop : loop_nld)
    hptc::assign_factor(drain_factors, fact_map, loop.size, loop.th_num,
        std::greater<GenNumType>());

  // Assign rest threads back to leading order loops
  hptc::assign_factor(drain_factors, fact_map, larger_loop.size,
      larger_loop.th_num, cmp_mod);
  hptc::assign_factor(drain_factors, fact_map, smaller_loop.size,
      smaller_loop.th_num, cmp_mod);
  fact_map.clear();
  drain_factors.clear();

  // Handle large prime factors
  if (large_primes.size() > 0) {
    const GenNumType rest_threads = std::accumulate(large_primes.begin(),
        large_primes.end(), 1, std::multiplies<GenNumType>());
    GenNumType rest_avail = 1;
    for (auto &loop : loop_nld)
      rest_avail *= loop.size;
    rest_avail *= larger_loop.size;
    if (not this->param_->is_common_leading())
      rest_avail *= smaller_loop.size;

    if (rest_avail <= rest_threads) {
      for (auto &loop : loop_nld) {
        loop.th_num *= loop.size;
        loop.size = 1;
      }
      larger_loop.th_num *= larger_loop.size;
      larger_loop.size = 1;
      smaller_loop.th_num *= smaller_loop.size;
      smaller_loop.size = 1;
    }
    else {
      auto avail_map = hptc::factorize(rest_avail);
      std::vector<GenNumType> avail_list;
      for (auto kv : avail_map)
        for (auto times = 0; times < kv.second; ++times)
          avail_list.push_back(kv.first);
      std::sort(avail_list.begin(), avail_list.end());

      auto compromise_factors = hptc::approx_prod(avail_list, rest_threads);
      fact_map.clear();
      for (auto factor : compromise_factors) {
        if (factor <= 1)
          continue;
        if (1 == fact_map.count(factor))
          ++fact_map[factor];
        else
          fact_map[factor] = 1;
      }

      for (auto &loop : loop_nld)
        hptc::assign_factor(drain_factors, fact_map, loop.size, loop.th_num,
            cmp_mod);
      hptc::assign_factor(drain_factors, fact_map, larger_loop.size,
          larger_loop.th_num, cmp_mod);
      hptc::assign_factor(drain_factors, fact_map, smaller_loop.size,
          smaller_loop.th_num, cmp_mod);
    }
  }

  // Parallelize the default strategy
  std::copy(this->parallel_template_.begin(), this->parallel_template_.end(),
      this->descriptor_.parallel_strategy.begin());
  this->descriptor_.parallel_strategy[in_ld_idx] *= loop_in_ld.th_num;
  this->descriptor_.parallel_strategy[out_ld_idx] *= loop_out_ld.th_num;
  for (auto &loop : loop_nld)
    this->descriptor_.parallel_strategy[loop.loop_idx] *= loop.th_num;

  // Update thread number
  this->threads_ = std::accumulate(this->descriptor_.parallel_strategy.begin(),
      this->descriptor_.parallel_strategy.end(), 1,
      std::multiplies<GenNumType>());
}


template <typename ParamType>
std::vector<LoopOrderTrans<ParamType::ORDER>>
PlanTransOptimizer<ParamType>::heur_loop_explorer_(
    const TensorIdx heur_num, TensorIdx tune_num) const {
  if (0 == heur_num)
    return { this->descriptor_.loop_order };
  tune_num = 0 == tune_num ? 1 : tune_num;

  // Create best heap to store auto tuning candidates
  // "Cost-Order" pair: (cost, loop order)
  using OrderDes = std::pair<double, LoopOrderTrans<ORDER>>;
  auto heap_cmp = [] (const auto &a, const auto &b) -> bool {
      return a.first < b.first; };

  // Create initial loop order
  LoopOrderTrans<ORDER> loop_order;
  for (TensorOrder order_idx = 0; order_idx < ORDER; ++order_idx)
    loop_order[order_idx] = order_idx;
  std::priority_queue<OrderDes, std::vector<OrderDes>, decltype(heap_cmp)>
      best_heap(heap_cmp);

  // Look for best
  const auto ld_idx = this->param_->begin_order_idx;
  TensorIdx times = 0;
  for (bool has_next = true; has_next and (times < heur_num or heur_num < 0);
      ++times, has_next = std::next_permutation(loop_order.begin() + ld_idx,
          loop_order.end())) {
    // Skip stride-1 leading loop in common leading case
    while (ld_idx == loop_order[0] and (has_next = std::next_permutation(
          loop_order.begin() + ld_idx, loop_order.end())));
    if (not has_next)
      break;

    // Update best heap
    auto new_cost = this->heur_loop_evaluator_(loop_order);
    if (tune_num < 0 or tune_num > best_heap.size())
      best_heap.push(OrderDes(new_cost, loop_order));
    else if (best_heap.top().first > new_cost) {
        best_heap.pop();
        best_heap.push(OrderDes(new_cost, loop_order));
    }
  }

  // Create result
  std::vector<LoopOrderTrans<ORDER>> result;
  result.reserve(best_heap.size());
  while (not best_heap.empty()) {
    result.push_back(best_heap.top().second);
    best_heap.pop();
  }

  return result;
}


template <typename ParamType>
double PlanTransOptimizer<ParamType>::heur_loop_evaluator_(
    const LoopOrderTrans<ORDER> &target_loop_order) const {
  // Locate begin index
  const auto merged_order = this->param_->merged_order;
  const auto begin_idx = this->param_->begin_order_idx;

  // Create loop penalty array
  // [..., 2 * penalty_step, 1 * penalty_step, 0 * penalty_step]
  std::vector<double> loop_penalty(merged_order, 0.0);
  loop_penalty.back() = this->penalty_begin;
  for (TensorIdx loop_idx = merged_order - 2; loop_idx >= 0; --loop_idx)
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


template <typename ParamType>
std::vector<ParaStrategyTrans<ParamType::ORDER>>
PlanTransOptimizer<ParamType>::heur_parallel_explorer_(
    const TensorIdx heur_num, TensorIdx tune_num) const {
  if (0 == heur_num)
    return { this->descriptor_.parallel_strategy };
  tune_num = 0 == tune_num ? 1 : tune_num;

  // Create best heap to store auto tuning candidates
  // "Cost-Parallel" pair: (cost, parallelization strategy)
  using ParaDes = std::pair<double, ParaStrategyTrans<ORDER>>;
  auto heap_cmp = [] (const auto &a, const auto &b) -> bool {
      return a.first < b.first; };

  // Create initial parallelization strategy.
  ParaStrategyTrans<ORDER> parallel_strategy;
  std::priority_queue<ParaDes, std::vector<ParaDes>, decltype(heap_cmp)>
      best_heap(heap_cmp);

  // Create stack for permuting all possible parallelization combinations
  const auto input_ld_idx = this->param_->begin_order_idx;
  TensorOrder fact_num = 0;
  std::stack<TensorOrder> loop_stack;
  std::vector<GenNumType> factors;
  auto fact_map = this->th_fact_map_;
  for (auto &fact : fact_map) {
    fact_num += fact.second;
    while (fact.second > 0) {
      factors.push_back(fact.first);
      --fact.second;
    }
  }
  for (TensorOrder loop_idx = 0; loop_idx < fact_num; ++loop_idx)
    loop_stack.push(input_ld_idx);

  // Closure for getting next permutation of parallelization
  auto get_next = [fact_num, input_ld_idx, &loop_stack] () -> bool {
    if (loop_stack.top() + 1 < ORDER)
      ++loop_stack.top();
    else {
      while (not loop_stack.empty() and loop_stack.top() + 1 == ORDER)
        loop_stack.pop();
      if (loop_stack.empty())
        return false;
      ++loop_stack.top();
      for (auto fact_idx = loop_stack.size(); fact_idx <= fact_num; ++fact_idx)
        loop_stack.push(input_ld_idx);
    }
    return true;
  };

  // Look for best
  TensorIdx times = 0;
  for (bool has_next = true; has_next and (times < heur_num or heur_num < 0);
      ++times, has_next = get_next()) {
    if (not has_next)
      break;

    // Get next parallelization
    std::vector<TensorOrder> pos;
    auto loop_stack_cp = loop_stack;
    while (not loop_stack_cp.empty()) {
      pos.push_back(loop_stack_cp.top());
      loop_stack_cp.pop();
    }
    parallel_strategy.fill(1);
    for (TensorOrder fact_idx = 0; fact_idx < fact_num; ++fact_idx)
      parallel_strategy[pos[fact_idx]] *= factors[fact_idx];

    // Update best heap
    auto new_cost = this->heur_parallel_evaluator_(parallel_strategy);
    if (tune_num < 0 or tune_num > best_heap.size())
      best_heap.push(ParaDes(new_cost, parallel_strategy));
    else if (best_heap.top().first > new_cost) {
        best_heap.pop();
        best_heap.push(ParaDes(new_cost, parallel_strategy));
    }
  }

  // Create result
  std::vector<ParaStrategyTrans<ORDER>> result;
  result.reserve(best_heap.size());
  while (not best_heap.empty()) {
    result.push_back(best_heap.top().second);
    best_heap.pop();
  }

  return result;
}


template <typename ParamType>
double PlanTransOptimizer<ParamType>::heur_parallel_evaluator_(
    const ParaStrategyTrans<ORDER> &target_para) const {
  // Find the biggest enabled kernel
  TensorIdx kn_idx = 0;
  while (this->descriptor_.description[0][kn_idx].is_disabled())
    ++kn_idx;
  const auto &target_loop = this->descriptor_.description[0][kn_idx];
  const auto begin_idx = this->param_->begin_order_idx;

  // Calculate costs
  double cost = 1.0;
  for (auto loop_idx = begin_idx; loop_idx < ORDER; ++loop_idx) {
    if (target_para[loop_idx] <= 1)
      continue;

    // Calculate number of steps assigned to each thread
    GenNumType steps_per_thread
        = (this->avail_parallel_[loop_idx] + target_para[loop_idx] - 1)
            / target_para[loop_idx];

    // Calculate load for loop at level loop_idx given certain number of threads
    const auto load = steps_per_thread * target_para[loop_idx];

    // Update cost
    cost *= static_cast<double>(load) / this->avail_parallel_[loop_idx];
  }

  // Strongly penalize parallelization at stride-1 loop in common leading case
  if (this->param_->is_common_leading())
    cost *= std::pow(this->penalty_factor_cl, target_para[begin_idx] - 1);

  // Penalize parallelization at input/output leading order loop
  cost *= std::pow(this->penalty_factor_inld,
      std::min(this->max_penalty_threads, target_para[begin_idx] - 1));
  cost *= std::pow(this->penalty_factor_outld,
      std::min(this->max_penalty_threads,
          target_para[this->param_->perm[begin_idx]] - 1));

  return cost;
}


template <typename ParamType>
std::vector<Descriptor<ParamType>>
PlanTransOptimizer<ParamType>::gen_candidates_(
    const std::vector<LoopOrderTrans<ORDER>> &loop_orders,
    const std::vector<ParaStrategyTrans<ORDER>> &parallel_strategies) const {
  std::vector<Descriptor<ParamType>> candidates;

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


template <typename ParamType>
void PlanTransOptimizer<ParamType>::parallelize_(
    Descriptor<ParamType> &descriptor) const {
  auto &des = descriptor.description;
  const auto &strategy = descriptor.parallel_strategy;
  const auto begin_loop_idx = this->param_->begin_order_idx;

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
    for (auto loop_idx = begin_loop_idx;
        loop_idx < ORDER and left_threads > 1; ++loop_idx) {
      // Compute step times at current loop level
      TensorIdx steps =
          (kn_basis.loop_end[loop_idx] - kn_basis.loop_begin[loop_idx]) /
          kn_basis.loop_step[loop_idx];

      // Create vector to store steps for each thread at current loop level
      const auto curr_para = strategy[loop_idx];
      std::vector<TensorIdx> split_steps(curr_para,
          curr_para <= steps ? steps / curr_para : 0);
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
