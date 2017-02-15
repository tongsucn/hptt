#pragma once
#ifndef HPTC_CGRAPH_CGRAPH_TRANS_TCC_
#define HPTC_CGRAPH_CGRAPH_TRANS_TCC_

template <typename ParamType,
          TensorOrder ORDER>
CGraphTrans<ParamType, ORDER>::CGraphTrans(
    const std::shared_ptr<ParamType> &param,
    const std::array<TensorOrder, ORDER> &loop_order,
    const std::vector<GenNumType> &strategy)
    : param_(param),
      loop_order_(loop_order),
      strategy_(strategy),
      operations_(nullptr) {
  // Compute thread number
  this->init_threads_();

  // Initialize operations
  this->init_operations_();

  // Compute vectorization
  this->init_vectorize_();

  // Set loop order
  this->init_loop_order_();

  // Set parallelization according to strategy and loop order
  this->init_parallel_();
}


template <typename ParamType,
          TensorOrder ORDER>
CGraphTrans<ParamType, ORDER>::~CGraphTrans() {
  this->release_operations_();
}


template <typename ParamType,
          TensorOrder ORDER>
INLINE void CGraphTrans<ParamType, ORDER>::operator()() {
#pragma omp parallel for
  for (TensorOrder idx = 0; idx < this->threads_; ++idx) {
    auto task = this->operations_ + idx;
    (*task)(this->param_->kn_fb);
    task = task->next;
    (*task)(this->param_->kn_fv);
    task = task->next;
    (*task)(this->param_->kn_fh);
    task = task->next;
    (*task)(this->param_->kn_fs);
    task = task->next;
    (*task)(this->param_->kn_hv);
    task = task->next;
    (*task)(this->param_->kn_hh);
    task = task->next;
    (*task)(this->param_->kn_hs);
    task = task->next;
    (*task)(this->param_->kn_sc);
    task = task->next;
    (*task)(this->param_->kn_sc);
  }
}


template <typename ParamType,
          TensorOrder ORDER>
void CGraphTrans<ParamType, ORDER>::init_operations_() {
  // Allocate memory for operations and set disable
  this->operations_ = new For_ [this->threads_];
  for (TensorIdx oper_idx = 0; oper_idx < this->threads_; ++oper_idx) {
    auto curr_oper = this->operations_ + oper_idx;
    curr_oper->init(this->param_);

    for (TensorIdx kn_idx = 0; kn_idx < 8; ++kn_idx) {
      curr_oper->next = new For_(this->param_);
      curr_oper = curr_oper->next;
    }
  }
}


template <typename ParamType,
          TensorOrder ORDER>
void CGraphTrans<ParamType, ORDER>::release_operations_() {
  // Release operations
  for (GenNumType idx = 0; idx < this->threads_; ++idx) {
    auto curr_oper = this->operations_[idx].next;
    while (nullptr != curr_oper) {
      auto next = curr_oper->next;
      delete curr_oper;
      curr_oper = next;
    }
  }

  delete [] this->operations_;
  this->operations_ = nullptr;
}


template <typename ParamType,
          TensorOrder ORDER>
void CGraphTrans<ParamType, ORDER>::init_vectorize_() {
  if (nullptr == this->operations_)
    return;

  // Check permutation type
  if (-1 == this->param_->perm_type()) {
    // For now, do nothing when leading dimensions are the same.
    return;
  }

  // Get parameters
  auto input_leading = this->param_->input_tensor.get_leading();
  auto output_leading = this->param_->output_tensor.get_leading();

  // Vectorize single thread version
  auto oper = &this->operations_[0];

  // Vectorization
  // Full big kernel (4 ncont x 4 cont full macro)
  auto fb_size = this->param_->kn_fb.get_cont_len();
  TensorOrder fb_cont_rest = input_leading, fb_ncont_rest = output_leading;
  TensorIdx fb_cont_begin = 0, fb_ncont_begin = 0;
  auto fb_set = this->init_kernel_(oper, fb_size, fb_size, fb_cont_rest,
      fb_ncont_rest, fb_cont_begin, fb_ncont_begin);

  // Full vertical kernel (4 ncont x 1 cont full macro)
  oper = oper->next;
  TensorOrder fv_cont_rest = fb_cont_rest, fv_ncont_rest = output_leading;
  TensorIdx fv_cont_begin = fb_cont_begin, fv_ncont_begin = 0;
  auto fv_set = this->init_kernel_(oper, this->param_->kn_fv.get_cont_len(),
      this->param_->kn_fv.get_ncont_len(), fv_cont_rest, fv_ncont_rest,
      fv_cont_begin, fv_ncont_begin);

  // Full horizontal kernel (1 ncont x 4 cont full macro)
  oper = oper->next;
  TensorOrder fh_cont_rest = input_leading, fh_ncont_rest = fb_ncont_rest;
  TensorIdx fh_cont_begin = 0, fh_ncont_begin = fb_ncont_begin;
  auto fh_set = this->init_kernel_(oper, this->param_->kn_fh.get_cont_len(),
      this->param_->kn_fh.get_ncont_len(), fh_cont_rest, fh_ncont_rest,
      fh_cont_begin, fh_ncont_begin);

  // Full small kernel (1 ncont x 1 cont full macro)
  oper = oper->next;
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
  auto fs_set = this->init_kernel_(oper, fs_size, fs_size, fs_cont_rest,
      fs_ncont_rest, fs_cont_begin, fs_ncont_begin);

  // Half vertical kernel (2 ncont x 1 cont half macro) and half horizontal
  // kernel (1 ncont x 2 cont half macro)
  oper = oper->next;
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
  auto hv_set = this->init_kernel_(oper, this->param_->kn_hv.get_cont_len(),
      this->param_->kn_hv.get_ncont_len(), hv_cont_rest, hv_ncont_rest,
      hv_cont_begin, hv_ncont_begin);

  oper = oper->next;
  auto hh_set = this->init_kernel_(oper, this->param_->kn_hh.get_cont_len(),
      this->param_->kn_hh.get_ncont_len(), hh_cont_rest, hh_ncont_rest,
      hh_cont_begin, hh_ncont_begin);

  // Half small kernel (1 ncont x 1 cont half macro)
  oper = oper->next;
  auto hs_size = this->param_->kn_hs.get_cont_len();
  if (hv_set and hh_set)
    ;
  else if (hv_set)
    hs_ncont_begin = hv_ncont_begin, hs_ncont_rest = hv_ncont_rest;
  else if (hh_set)
    hs_ncont_begin = hh_ncont_begin, hs_ncont_rest = hh_ncont_rest;
  auto hs_set = this->init_kernel_(oper, hs_size, hs_size, hs_cont_rest,
      hs_ncont_rest, hs_cont_begin, hs_ncont_begin);

  // Scalar horizontal and then scalar vertical
  oper = oper->next;
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
  this->init_kernel_(oper, 1, 1, sh_cont_rest, sh_ncont_rest, sh_cont_begin,
      sh_ncont_begin);

  oper = oper->next;
  this->init_kernel_(oper, 1, 1, sv_cont_rest, sv_ncont_rest, sv_cont_begin,
      sv_ncont_begin);
}


template <typename ParamType,
          TensorOrder ORDER>
bool CGraphTrans<ParamType, ORDER>::init_kernel_(For_ *oper,
    GenNumType cont_len, GenNumType ncont_len, TensorOrder &cont_rest,
    TensorOrder &ncont_rest, TensorIdx &cont_begin, TensorIdx &ncont_begin) {
  if (cont_len <= cont_rest and ncont_len <= ncont_rest) {
    const TensorIdx begin_idx = ORDER - this->param_->merged_order;
    const TensorIdx ncont_rest_idx = begin_idx + this->param_->perm[begin_idx];

    // Set pass on merged orders
    oper->set_pass(begin_idx);
    // Vectorize
    for (TensorIdx idx = begin_idx; idx < ORDER; ++idx) {
      if (begin_idx == idx) {
        // Vectorize input tensor stride-1 order
        TensorIdx times = cont_rest / cont_len;
        oper->begin(idx) = cont_begin;

        auto span = times * cont_len;
        cont_begin += span;
        cont_rest -= span;

        oper->end(idx) = cont_begin;
        oper->step(idx) = cont_len;
      }
      else if (ncont_rest_idx == idx) {
        // Vectorize output tensor stride-1 order
        TensorIdx times = ncont_rest / ncont_len;
        oper->begin(idx) = ncont_begin;

        auto span = times * ncont_len;
        ncont_begin += span;
        ncont_rest -= span;

        oper->end(idx) = ncont_begin;
        oper->step(idx) = ncont_len;
      }
      else {
        oper->begin(idx) = 0;
        oper->end(idx) = this->param_->input_tensor.get_size()[idx];
        oper->step(idx) = 1;
      }
    }

    return true;
  }

  return false;
}


template <typename ParamType,
          TensorOrder ORDER>
void CGraphTrans<ParamType, ORDER>::init_loop_order_() {
  // Set order of loops
  for (TensorIdx idx = 0; idx < this->threads_; ++idx) {
    this->operations_[idx].set_order(this->loop_order_);
    auto next = this->operations_[idx].next;
    while (nullptr != next) {
      next->set_order(this->loop_order_);
      next = next->next;
    }
  }
}


template <typename ParamType,
          TensorOrder ORDER>
void CGraphTrans<ParamType, ORDER>::init_threads_() {
  // Compute product of parallelization strategy
  this->threads_ = std::accumulate(this->strategy_.begin(),
      this->strategy_.end(), 1, std::multiplies<GenNumType>());
  // If product is zero, using single thread
  if (0 == this->threads_)
    this->threads_ = 1;
}


template <typename ParamType,
          TensorOrder ORDER>
void CGraphTrans<ParamType, ORDER>::init_parallel_() {
  // This function assumes parallelization strategy is correct,
  // ill-formed strategy will lead to undefined behavior.
  if (this->threads_ <= 1)
    return;

  // Prepare
  std::vector<For_ *> oper_ptr;
  for (TensorIdx idx = 0; idx < this->threads_; ++idx)
    oper_ptr.push_back(this->operations_ + idx);

  // Locate begin order for parallelization in case order merge
  const auto begin_idx
      = static_cast<TensorIdx>(ORDER - this->param_->merged_order);
  const auto end_idx
      = begin_idx + static_cast<TensorIdx>(this->strategy_.size());

  // Parallelize
  for (; nullptr != oper_ptr[0]; std::for_each(oper_ptr.begin(), oper_ptr.end(),
        [] (auto &ptr) -> void { ptr = ptr->next; })) {
    // Skip disabled kernel
    if (oper_ptr[0]->is_disable())
      continue;

    // Copy from vectorized kernel
    for (GenNumType idx = 1; idx < this->threads_; ++idx)
      *oper_ptr[idx] = *oper_ptr[0];

    auto assign_threads = this->threads_;
    for (auto order_idx = begin_idx; order_idx < end_idx; ++order_idx) {
      // Locate loop
      auto loop_idx = this->loop_order_[order_idx];

      // Assign steps at loop level order_idx to threads
      GenNumType steps =
          (oper_ptr[0]->end(loop_idx) - oper_ptr[0]->begin(loop_idx)) /
          oper_ptr[0]->step(loop_idx);

      // Create split step vector
      auto curr_para = this->strategy_[order_idx - begin_idx];
      std::vector<TensorIdx> split_steps(curr_para, steps / curr_para);

      // Deal with rest steps
      auto rest_steps = steps % curr_para;
      std::for_each(split_steps.end() - rest_steps, split_steps.end(),
          [] (auto &num) { ++num; });

      // Create unit spans
      std::vector<TensorIdx> unit_begins(assign_threads),
          unit_ends(assign_threads);
      GenNumType copies = assign_threads / curr_para;
      for (GenNumType cp_idx = 0, begin_val = oper_ptr[0]->begin(loop_idx);
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
      std::vector<GenNumType> begins(this->threads_), ends(this->threads_);
      for (GenNumType offset = 0; offset < this->threads_;
          offset += assign_threads) {
        std::copy(unit_begins.begin(), unit_begins.end(),
            begins.begin() + offset);
        std::copy(unit_ends.begin(), unit_ends.end(), ends.begin() + offset);
      }

      for (GenNumType oper_idx = 0; oper_idx < this->threads_; ++oper_idx) {
        oper_ptr[oper_idx]->begin(loop_idx) = begins[oper_idx];
        oper_ptr[oper_idx]->end(loop_idx) = ends[oper_idx];
        oper_ptr[oper_idx]->step(loop_idx) = oper_ptr[0]->step(loop_idx);
      }

      assign_threads /= curr_para;
    }
  }
}

#endif // HPTC_CGRAPH_CGRAPH_TRANS_TCC_
