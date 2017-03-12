#pragma once
#ifndef HPTC_CGRAPH_CGRAPH_TRANS_TCC_
#define HPTC_CGRAPH_CGRAPH_TRANS_TCC_

template <TensorOrder ORDER>
CGraphTransDescriptor<ORDER>::CGraphTransDescriptor()
    : parallel_strategy{ 1 },
      description(1) {
}


template <typename ParamType,
          TensorOrder ORDER>
CGraphTrans<ParamType, ORDER>::~CGraphTrans() {
  this->release_operations_();
}


template <typename ParamType,
          TensorOrder ORDER>
INLINE void CGraphTrans<ParamType, ORDER>::operator()() {
  if (not this->param_->is_common_leading())
    this->exec_general_();
  else
    this->exec_common_leading_();
}


template <typename ParamType,
          TensorOrder ORDER>
CGraphTrans<ParamType, ORDER>::CGraphTrans(
    const std::shared_ptr<ParamType> &param,
    const CGraphTransDescriptor<ORDER> &descriptor)
    : param_(param),
      descriptor_(descriptor),
      threads_(descriptor.description.size()),
      operations_(this->threads_ > 0 ? new For_ [this->threads_] : nullptr) {
  this->init_();
}


template <typename ParamType,
          TensorOrder ORDER>
void CGraphTrans<ParamType, ORDER>::init_() {
  // Initialize for loops' parameters and loop order
  for (GenNumType th_idx = 0, idx_end = this->descriptor_.description[0].size();
      th_idx < this->threads_; ++th_idx) {
    auto curr_oper = this->operations_ + th_idx;
    curr_oper->init(this->descriptor_.loop_order,
        this->descriptor_.description[th_idx][0],
        this->param_->begin_order_idx, this->param_->perm);

    for (GenNumType kn_idx = 1; kn_idx < idx_end; ++kn_idx) {
      curr_oper->next = new For_(this->descriptor_.loop_order,
          this->descriptor_.description[th_idx][kn_idx],
          this->param_->begin_order_idx, this->param_->perm);
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
INLINE void CGraphTrans<ParamType, ORDER>::exec_general_() {
#pragma omp parallel for schedule(static)
  for (TensorOrder idx = 0; idx < this->threads_; ++idx) {
    auto task = this->operations_ + idx;
    (*task)(this->param_->kn_fb, this->param_->input_tensor,
        this->param_->output_tensor, this->param_->input_stride,
        this->param_->output_stride);

    task = task->next;
    (*task)(this->param_->kn_fv, this->param_->input_tensor,
        this->param_->output_tensor, this->param_->input_stride,
        this->param_->output_stride);

    task = task->next;
    (*task)(this->param_->kn_fh, this->param_->input_tensor,
        this->param_->output_tensor, this->param_->input_stride,
        this->param_->output_stride);

    task = task->next;
    (*task)(this->param_->kn_fs, this->param_->input_tensor,
        this->param_->output_tensor, this->param_->input_stride,
        this->param_->output_stride);

    task = task->next;
    (*task)(this->param_->kn_hv, this->param_->input_tensor,
        this->param_->output_tensor, this->param_->input_stride,
        this->param_->output_stride);

    task = task->next;
    (*task)(this->param_->kn_hh, this->param_->input_tensor,
        this->param_->output_tensor, this->param_->input_stride,
        this->param_->output_stride);

    task = task->next;
    (*task)(this->param_->kn_hs, this->param_->input_tensor,
        this->param_->output_tensor, this->param_->input_stride,
        this->param_->output_stride);

    task = task->next;
    (*task)(this->param_->kn_ln, this->param_->input_tensor,
        this->param_->output_tensor, 1, 0);

    task = task->next;
    (*task)(this->param_->kn_ln, this->param_->input_tensor,
        this->param_->output_tensor, 1, 0);
  }
}


template <typename ParamType,
          TensorOrder ORDER>
INLINE void CGraphTrans<ParamType, ORDER>::exec_common_leading_() {
  // Get leading length
  const auto ld_len = static_cast<TensorIdx>(this->param_->get_leading().first);

#pragma omp parallel for schedule(static)
  for (TensorOrder idx = 0; idx < this->threads_; ++idx)
    this->operations_[idx](this->param_->kn_ln, this->param_->input_tensor,
        this->param_->output_tensor, ld_len, 0);
}

#endif // HPTC_CGRAPH_CGRAPH_TRANS_TCC_
