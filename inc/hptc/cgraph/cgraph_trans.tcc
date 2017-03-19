#pragma once
#ifndef HPTC_CGRAPH_CGRAPH_TRANS_TCC_
#define HPTC_CGRAPH_CGRAPH_TRANS_TCC_

template <typename ParamType>
CGraphTrans<ParamType>::CGraphTransDescriptor::CGraphTransDescriptor()
    : description(1) {
  for (TensorOrder order_idx = 0; order_idx < ORDER; ++order_idx)
    this->loop_order[order_idx] = order_idx;
  this->parallel_strategy.fill(1);
  this->description[0].fill(LoopParamTrans<ORDER>());
}


template <typename ParamType>
CGraphTrans<ParamType>::~CGraphTrans() {
  this->release_operations_();
}


template <typename ParamType>
INLINE void CGraphTrans<ParamType>::operator()() {
  if (not this->param_->is_common_leading())
    this->exec_general_();
  else
    this->exec_common_leading_();
}


template <typename ParamType>
INLINE CGraphTrans<ParamType>::CGraphTransDescriptor
CGraphTrans<ParamType>::get_descriptor() const {
  return this->descriptor_;
}


template <typename ParamType>
CGraphTrans<ParamType>::CGraphTrans(const std::shared_ptr<ParamType> &param,
    const CGraphTransDescriptor &descriptor)
    : param_(param),
      descriptor_(descriptor),
      threads_(descriptor.description.size()),
      operations_(this->threads_ > 0 ? new For_ [this->threads_] : nullptr) {
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


template <typename ParamType>
void CGraphTrans<ParamType>::release_operations_() {
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


template <typename ParamType>
INLINE void CGraphTrans<ParamType>::exec_general_() {
#pragma omp parallel for schedule(static)
  for (TensorOrder idx = 0; idx < this->threads_; ++idx) {
    auto task = this->operations_ + idx;
    (*task)(this->param_->kn.knf_1x1, this->param_->input_tensor,
        this->param_->output_tensor, this->param_->input_stride,
        this->param_->output_stride);

    task = task->next;
    (*task)(this->param_->kn.knf_1x2, this->param_->input_tensor,
        this->param_->output_tensor, this->param_->input_stride,
        this->param_->output_stride);

    task = task->next;
    (*task)(this->param_->kn.knf_1x3, this->param_->input_tensor,
        this->param_->output_tensor, this->param_->input_stride,
        this->param_->output_stride);

    task = task->next;
    (*task)(this->param_->kn.knf_1x4, this->param_->input_tensor,
        this->param_->output_tensor, this->param_->input_stride,
        this->param_->output_stride);

    task = task->next;
    (*task)(this->param_->kn.knf_2x1, this->param_->input_tensor,
        this->param_->output_tensor, this->param_->input_stride,
        this->param_->output_stride);

    task = task->next;
    (*task)(this->param_->kn.knf_2x2, this->param_->input_tensor,
        this->param_->output_tensor, this->param_->input_stride,
        this->param_->output_stride);

    task = task->next;
    (*task)(this->param_->kn.knf_2x3, this->param_->input_tensor,
        this->param_->output_tensor, this->param_->input_stride,
        this->param_->output_stride);

    task = task->next;
    (*task)(this->param_->kn.knf_2x4, this->param_->input_tensor,
        this->param_->output_tensor, this->param_->input_stride,
        this->param_->output_stride);

    task = task->next;
    (*task)(this->param_->kn.knf_3x1, this->param_->input_tensor,
        this->param_->output_tensor, this->param_->input_stride,
        this->param_->output_stride);

    task = task->next;
    (*task)(this->param_->kn.knf_3x2, this->param_->input_tensor,
        this->param_->output_tensor, this->param_->input_stride,
        this->param_->output_stride);

    task = task->next;
    (*task)(this->param_->kn.knf_3x3, this->param_->input_tensor,
        this->param_->output_tensor, this->param_->input_stride,
        this->param_->output_stride);

    task = task->next;
    (*task)(this->param_->kn.knf_3x4, this->param_->input_tensor,
        this->param_->output_tensor, this->param_->input_stride,
        this->param_->output_stride);

    task = task->next;
    (*task)(this->param_->kn.knf_4x1, this->param_->input_tensor,
        this->param_->output_tensor, this->param_->input_stride,
        this->param_->output_stride);

    task = task->next;
    (*task)(this->param_->kn.knf_4x2, this->param_->input_tensor,
        this->param_->output_tensor, this->param_->input_stride,
        this->param_->output_stride);

    task = task->next;
    (*task)(this->param_->kn.knf_4x3, this->param_->input_tensor,
        this->param_->output_tensor, this->param_->input_stride,
        this->param_->output_stride);

    task = task->next;
    (*task)(this->param_->kn.knf_4x4, this->param_->input_tensor,
        this->param_->output_tensor, this->param_->input_stride,
        this->param_->output_stride);

    task = task->next;
    (*task)(this->param_->kn.knh_1x1, this->param_->input_tensor,
        this->param_->output_tensor, this->param_->input_stride,
        this->param_->output_stride);

    task = task->next;
    (*task)(this->param_->kn.knh_1x2, this->param_->input_tensor,
        this->param_->output_tensor, this->param_->input_stride,
        this->param_->output_stride);

    task = task->next;
    (*task)(this->param_->kn.knh_1x3, this->param_->input_tensor,
        this->param_->output_tensor, this->param_->input_stride,
        this->param_->output_stride);

    task = task->next;
    (*task)(this->param_->kn.knh_1x4, this->param_->input_tensor,
        this->param_->output_tensor, this->param_->input_stride,
        this->param_->output_stride);

    task = task->next;
    (*task)(this->param_->kn.knh_2x1, this->param_->input_tensor,
        this->param_->output_tensor, this->param_->input_stride,
        this->param_->output_stride);

    task = task->next;
    (*task)(this->param_->kn.knh_3x1, this->param_->input_tensor,
        this->param_->output_tensor, this->param_->input_stride,
        this->param_->output_stride);

    task = task->next;
    (*task)(this->param_->kn.knh_4x1, this->param_->input_tensor,
        this->param_->output_tensor, this->param_->input_stride,
        this->param_->output_stride);

    task = task->next;
    (*task)(this->param_->kn.kn_lin, this->param_->input_tensor,
        this->param_->output_tensor, 1, 0);

    task = task->next;
    (*task)(this->param_->kn.kn_lin, this->param_->input_tensor,
        this->param_->output_tensor, 1, 0);
  }
}


template <typename ParamType>
INLINE void CGraphTrans<ParamType>::exec_common_leading_() {
  const auto ld_len = static_cast<TensorIdx>(this->param_->get_leading().first);
#pragma omp parallel for schedule(static)
  for (TensorOrder idx = 0; idx < this->threads_; ++idx)
    this->operations_[idx](this->param_->kn.kn_lin, this->param_->input_tensor,
        this->param_->output_tensor, ld_len, 0);
}


/*
 * Avoid template instantiation for class CGraphTrans, import generated extern
 * template declaration.
 */
#include "cgraph_trans_gen.tcc"

#endif // HPTC_CGRAPH_CGRAPH_TRANS_TCC_
