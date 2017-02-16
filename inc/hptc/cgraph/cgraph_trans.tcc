#pragma once
#ifndef HPTC_CGRAPH_CGRAPH_TRANS_TCC_
#define HPTC_CGRAPH_CGRAPH_TRANS_TCC_

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
    const LoopOrder<ORDER> loop_order,
    const CGraphTransDescriptor<ORDER> &descriptor)
    : param_(param),
      descriptor(descriptor),
      threads_(descriptor.size()),
      loop_order_(loop_order),
      operations_(0 != this->threads_ ? new For_ [this->threads_] : nullptr) {
  this->init_(descriptor);
}


template <typename ParamType,
          TensorOrder ORDER>
void CGraphTrans<ParamType, ORDER>::init_(
    const CGraphTransDescriptor<ORDER> &descriptor) {
  // Initialize for loops' parameters and loop order
  for (TensorIdx oper_idx = 0; oper_idx < this->threads_; ++oper_idx) {
    auto curr_oper = this->operations_ + oper_idx;
    curr_oper->init(this->param_);
    curr_oper->set_loop(descriptor[oper_idx][0]);
    curr_oper->set_order(this->loop_order_);

    const GenNumType idx_end = descriptor[0].size();
    for (GenNumType kn_idx = 1; kn_idx < idx_end; ++kn_idx) {
      curr_oper->next = new For_(this->param_);
      curr_oper = curr_oper->next;
      curr_oper->set_loop(descriptor[oper_idx][kn_idx]);
      curr_oper->set_order(this->loop_order_);
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
