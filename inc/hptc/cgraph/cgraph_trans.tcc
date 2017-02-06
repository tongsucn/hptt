#pragma once
#ifndef HPTC_CGRAPH_CGRAPH_TRANS_TCC_
#define HPTC_CGRAPH_CGRAPH_TRANS_TCC_

template <typename ParamType,
          TensorOrder ORDER>
CGraphTrans<ParamType, ORDER>::CGraphTrans(
    const std::shared_ptr<ParamType> &param, GenNumType threads)
    : param_(param),
      threads_(0 == threads ? std::thread::hardware_concurrency() : threads_),
      operations_(nullptr) {
  // If hardware concurrency is not well defined, then use single thread
  this->threads_ = 0 == this->threads_ ? 1 : this->threads_;

  // Allocate memory for operations
  this->operations_ = new For_ [this->threads_];
  for (TensorIdx oper_idx = 0; oper_idx < this->threads_; ++oper_idx) {
    auto curr_oper = this->operations_ + oper_idx;
    for (TensorIdx kn_idx = 0; kn_idx < 8; ++kn_idx) {
      curr_oper->next = new For_;
      curr_oper = curr_oper->next;
    }
  }

  // Vectorization
  this->init_vectorize_();
}


/*template <typename ParamType,
          TensorOrder ORDER>
CGraphTrans<ParamType, ORDER>::CGraphTrans(const CGraphTrans &graph)
    : param_(graph.param_),
      threads_(graph.threads_),
      operations_(0 == graph.threads_ ? nullptr : new For_ [this->threads_]) {
  if (nullptr == this->operations_)
    return;

  for (TensorIdx idx = 0; idx < this->threads_; ++idx) {
    this->operations_[idx] = graph.operations_[idx];
    auto curr = &this->operations_[idx];
    auto next = graph.operations_[idx].next;
    while (nullptr != next) {
      curr->next = new For_(*next);
      next = next->next;
      curr = curr->next;
      curr->next = nullptr;
    }
  }
}


template <typename ParamType,
          TensorOrder ORDER>
CGraphTrans<ParamType, ORDER> &CGraphTrans<ParamType, ORDER>::operator=(
    const CGraphTrans &graph) {
  this->param_ = graph.param_;
  this->threads_ = graph.threads_;

  // Release operations before copy
  this->release_operations_();

  if (0 == this->threads_) {
    this->operations_ = nullptr;
    return *this;
  }

  for (TensorIdx idx = 0; idx < this->threads_; ++idx) {
    this->operations_[idx] = graph.operations_[idx];
    auto curr = &this->operations_[idx];
    auto next = graph.operations_[idx].next;
    while (nullptr != next) {
      curr->next = new For_(*next);
      next = next->next;
      curr = curr->next;
      curr->next = nullptr;
    }
  }

  return *this;
}*/


template <typename ParamType,
          TensorOrder ORDER>
CGraphTrans<ParamType, ORDER>::~CGraphTrans() {
  this->release_operations_();
}


template <typename ParamType,
          TensorOrder ORDER>
TensorIdx CGraphTrans<ParamType, ORDER>::set_loop_order(
    const std::vector<TensorOrder> &order) {
  // If order's size does not equal to tensor's order, return -1.
  if (ORDER != order.size())
    return -1;

  // Check if the input vector's values are correct. Return -2 if incorrect.
  std::vector<bool> check_map(ORDER, true);
  for (auto val : order)
    if (order >= ORDER)
      return -2;
    else
      check_map[val] = false;

  for (TensorIdx idx = 0; idx < ORDER; ++idx)
    if (check_map[idx])
      return -2;

  // Set order of loops
  for (TensorIdx idx = 0; idx < this->threads_; ++idx) {
    this->operations_[idx].set_order(order);
    auto next = this->operations_[idx].next;
    while (nullptr != next) {
      next->set_order(order);
      next = next->next;
    }
  }

  return 0;
}


template <typename ParamType,
          TensorOrder ORDER>
TensorIdx CGraphTrans<ParamType, ORDER>::set_parallel(
    const std::vector<GenNumType> &strategy, const GenNumType threads) {
}


template <typename ParamType,
          TensorOrder ORDER>
INLINE TensorIdx CGraphTrans<ParamType, ORDER>::exec() {
  std::thread *thread_pool = new std::thread [this->threads_];
  for (TensorOrder idx = 0; idx < ORDER; ++idx)
    thread_pool[idx] = std::thread(this->thread_task_);

  for (TensorOrder idx = 0; idx < ORDER; ++idx)
    thread_pool[idx].join();

  delete [] thread_pool;

  return 0;
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

  // Only initialize single thread version at this time.
  auto oper = &this->operations_[0];
  for (TensorOrder idx = 1; idx < this->threads_; ++idx)
    this->operations_[idx].set_disable();

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
    fs_cont_begin = fb_cont_begin, fs_ncont_begin = fs_ncont_begin;
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
    hv_cont_begin = fv_cont_begin, hh_ncont_begin_fv_ncont_begin;
    hs_cont_rest = fv_cont_rest, hs_ncont_rest = fv_ncont_rest;
    hs_cont_begin = fv_cont_begin, hs_ncont_begin = fv_ncont_begin;
  }
  else if (fh_set) {
    hv_cont_rest = fh_cont_rest, hh_ncont_rest = fh_ncont_rest;
    hv_cont_begin = fh_cont_begin, hh_ncont_begin_fh_ncont_begin;
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

  // Scalar vertical and scalar horizontal
  oper = oper->next;
  if (hs_set) {
    ;
  }
  this->init_kernel_(oper, 1, 1, input_leading, output_leading, input_begin,
      output_begin);

  // Scalar horizontal
  oper = oper->next;
  this->init_kernel_(oper, 1, 1, input_leading, output_leading, input_begin,
      output_begin);
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
        TensorIdx times = cont_rest / cont_len;
        oper->set_begin(cont_begin, idx);

        auto span = times * cont_len;
        cont_begin += span;
        cont_rest -= span;

        oper->set_end(cont_begin, idx);
        oper->set_step(cont_len, idx);
      }
      else if (ncont_rest_idx == idx) {
        TensorIdx times = ncont_rest / ncont_len;
        oper->set_begin(ncont_begin, idx);

        auto span = times * ncont_len;
        ncont_begin += span;
        ncont_rest -= span;

        oper->set_end(times * ncont_len, idx);
        oper->set_step(ncont_len, idx);
      }
      else {
        oper->set_begin(0, idx);
        oper->set_end(this->param_->input_tensor.get_size()[idx], idx);
        oper->set_step(1, idx);
      }
    }

    return true;
  }
  else {
    oper->set_disable();
    return false;
  }
}


template <typename ParamType,
          TensorOrder ORDER>
void CGraphTrans<ParamType, ORDER>::init_parallelize_() {
}


template <typename ParamType,
          TensorOrder ORDER>
INLINE void CGraphTrans<ParamType, ORDER>::thread_task_(TensorOrder idx) {
  auto task = &this->operations_[idx];
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


template <typename ParamType,
          TensorOrder ORDER>
void CGraphTrans<ParamType, ORDER>::release_operations_() {
  for (GenNumType idx = 0; idx < this->threads_; ++idx) {
    auto curr_oper = this->operations_[idx].next;
    while (nullptr != curr_oper) {
      auto next = curr_oper->next;
      delete curr_oper;
      curr_oper = next;
    }
  }

  this->threads_ = 0;
  delete [] this->operations_;
  this->operations_ = nullptr;
}

#endif // HPTC_CGRAPH_CGRAPH_TRANS_TCC_
