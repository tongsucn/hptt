#pragma once
#ifndef HPTC_CGRAPH_CGRAPH_TRANS_TCC_
#define HPTC_CGRAPH_CGRAPH_TRANS_TCC_

template <TensorOrder ORDER>
CGraphTransDescriptor<ORDER>::CGraphTransDescriptor(GenNumType thread_num)
    : description(thread_num) {
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
CGraphTrans<ParamType, ORDER>::CGraphTrans(
    const std::shared_ptr<ParamType> &param,
    const CGraphTransDescriptor<ORDER> &descriptor)
    : param_(param),
      descriptor_(descriptor),
      threads_(descriptor.description.size()),
      operations_(0 != this->threads_ ? new For_ [this->threads_] : nullptr) {
  this->init_();
}


template <typename ParamType,
          TensorOrder ORDER>
void CGraphTrans<ParamType, ORDER>::init_() {
  // Initialize for loops' parameters and loop order
  for (GenNumType th_idx = 0, idx_end = this->descriptor_.description[0].size();
      th_idx < this->threads_; ++th_idx) {
    auto curr_oper = this->operations_ + th_idx;
    curr_oper->init(this->param_, this->descriptor_.loop_order,
        this->descriptor_.description[th_idx][0]);

    for (GenNumType kn_idx = 1; kn_idx < idx_end; ++kn_idx) {
      curr_oper->next = new For_(this->param_, this->descriptor_.loop_order,
          this->descriptor_.description[th_idx][kn_idx]);
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

#endif // HPTC_CGRAPH_CGRAPH_TRANS_TCC_
