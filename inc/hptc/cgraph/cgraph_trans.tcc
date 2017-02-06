#pragma once
#ifndef HPTC_CGRAPH_CGRAPH_TRANS_TCC_
#define HPTC_CGRAPH_CGRAPH_TRANS_TCC_

template <typename ParamType,
          TensorOrder ORDER>
CGraphTrans<ParamType, ORDER>::CGraphTrans(
    const std::shared_ptr<ParamType> &param,
    const std::vector<TensorOrder> &loop_order,
    const std::vector<GenNumType> &para_strategy)
    : param_(param),
      thread_pool_(nullptr),
      operations_(nullptr) {
  // Compute thread number
  // Compute product of parallelization strategy
  this->threads_ = this->para_strategy_ = std::accumulate(para_strategy.begin(),
      para_strategy.end(), 1, std::multiplies<GenNumType>());
  // If product is zero, using hardware concurrency
  if (0 == this->para_strategy_)
    this->threads_ = std::thread::hardware_concurrency();
  // If hardware concurrency is not well defined, then use single thread
  if (0 == this->threads_)
    this->threads_ = 1;

  // Initialize operations
  this->init_operations_();

  // Compute vectorization
  this->init_vectorize_();

  // Set loop order
  this->init_loop_order_(loop_order);

  // Set parallelization according to strategy and loop order
  this->init_parallel_(loop_order, para_strategy);
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
INLINE void CGraphTrans<ParamType, ORDER>::operator()() {
  for (TensorOrder idx = 0; idx < this->threads_; ++idx)
    this->thread_pool_[idx] = std::thread(&CGraphTrans::task_, this, idx);

  for (TensorOrder idx = 0; idx < this->threads_; ++idx)
    this->thread_pool_[idx].join();
}


template <typename ParamType,
          TensorOrder ORDER>
void CGraphTrans<ParamType, ORDER>::init_operations_() {
  // Initialize thread utilities
  this->thread_pool_ = new std::thread [this->threads_];

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
  // Release threads
  delete [] this->thread_pool_;
  this->thread_pool_ = nullptr;

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

        oper->set_end(ncont_begin, idx);
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
  else
    return false;
}


template <typename ParamType,
          TensorOrder ORDER>
void CGraphTrans<ParamType, ORDER>::init_loop_order_(
    const std::vector<TensorOrder> &order) {
  // Set order of loops
  for (TensorIdx idx = 0; idx < this->threads_; ++idx) {
    this->operations_[idx].set_order(order);
    auto next = this->operations_[idx].next;
    while (nullptr != next) {
      next->set_order(order);
      next = next->next;
    }
  }
}


template <typename ParamType,
          TensorOrder ORDER>
void CGraphTrans<ParamType, ORDER>::init_parallel_(
    const std::vector<TensorOrder> &loop_order,
    const std::vector<GenNumType> &para_strategy) {
  if (1 == this->threads_)
    return;

  auto begin_order = ORDER - this->param_->merged_order;
}


template <typename ParamType,
          TensorOrder ORDER>
INLINE void CGraphTrans<ParamType, ORDER>::task_(TensorOrder idx) {
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

#endif // HPTC_CGRAPH_CGRAPH_TRANS_TCC_
