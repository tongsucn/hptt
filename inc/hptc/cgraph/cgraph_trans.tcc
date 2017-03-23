#pragma once
#ifndef HPTC_CGRAPH_CGRAPH_TRANS_TCC_
#define HPTC_CGRAPH_CGRAPH_TRANS_TCC_

/*
 * Implementation for class CGraphTrans::Descriptor
 */
template <typename ParamType>
CGraphTrans<ParamType>::Descriptor::Descriptor()
    : description(1) {
  for (auto order_idx = 0; order_idx < ORDER; ++order_idx)
    this->loop_order[order_idx] = order_idx;
  this->parallel_strategy.fill(1);
  this->description[0].fill(LoopParamTrans<ORDER>());
}


/*
 * Implementation for class CGraphTrans
 */
template <typename ParamType>
CGraphTrans<ParamType>::~CGraphTrans() {
  this->release_();
}


template <typename ParamType>
INLINE void CGraphTrans<ParamType>::exec() {
  if (not this->param_->is_common_leading())
    this->exec_general_();
  else
    this->exec_common_leading_();
}


template <typename ParamType>
INLINE void CGraphTrans<ParamType>::operator()() {
  this->exec();
}


template <typename ParamType>
INLINE CGraphTrans<ParamType>::Descriptor
CGraphTrans<ParamType>::get_descriptor() const {
  return this->descriptor_;
}


template <typename ParamType>
CGraphTrans<ParamType>::CGraphTrans(const std::shared_ptr<ParamType> &param,
    const Descriptor &descriptor)
    : param_(param),
      threads_(0),
      operations_(nullptr) {
  this->init(descriptor);
}


template <typename ParamType>
void CGraphTrans<ParamType>::init(const Descriptor &descriptor) {
  // Release previous (if exists) before initialization
  this->release_();

  // Initialize members
  this->threads_ = descriptor.description.size();
  if (this->threads_ < 1)
    return;
  this->descriptor_ = descriptor;
  this->operations_ = new For_ [this->threads_];

  // Initialize for loops' parameters and loop order
  for (TensorUInt th_idx = 0, idx_end = this->descriptor_.description[0].size();
      th_idx < this->threads_; ++th_idx) {
    auto curr_oper = this->operations_ + th_idx;
    curr_oper->init(this->descriptor_.loop_order,
        this->descriptor_.description[th_idx][0],
        this->param_->begin_order_idx, this->param_->perm);

    for (auto kn_idx = 1; kn_idx < idx_end; ++kn_idx) {
      curr_oper->next = new For_(this->descriptor_.loop_order,
          this->descriptor_.description[th_idx][kn_idx],
          this->param_->begin_order_idx, this->param_->perm);
      curr_oper = curr_oper->next;
    }
  }
}


template <typename ParamType>
void CGraphTrans<ParamType>::release_() {
  // Release operations
  for (auto idx = 0; idx < this->threads_; ++idx) {
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


template <typename ParamType>
INLINE void CGraphTrans<ParamType>::exec_general_() {
  const auto &kn = this->param_->kn;
  auto &input_tensor = this->param_->input_tensor;
  auto &output_tensor = this->param_->output_tensor;
  const auto input_stride = this->param_->input_stride;
  const auto output_stride = this->param_->output_stride;
  const auto &reg_alpha_full = this->param_->reg_alpha_full;
  const auto &reg_beta_full = this->param_->reg_beta_full;
  const auto &reg_alpha_half = this->param_->reg_alpha_half;
  const auto &reg_beta_half = this->param_->reg_beta_half;
  const auto &reg_alpha_linear = this->param_->reg_alpha_linear;
  const auto &reg_beta_linear = this->param_->reg_beta_linear;

#pragma omp parallel for schedule(static)
  for (auto th_idx = 0; th_idx < this->threads_; ++th_idx) {
    auto task = this->operations_ + th_idx;
    (*task)(kn.knf_1x1, input_tensor, output_tensor, input_stride,
        output_stride, reg_alpha_full, reg_beta_full);

    task = task->next;
    (*task)(kn.knf_1x2, input_tensor, output_tensor, input_stride,
        output_stride, reg_alpha_full, reg_beta_full);

    task = task->next;
    (*task)(kn.knf_1x3, input_tensor, output_tensor, input_stride,
        output_stride, reg_alpha_full, reg_beta_full);

    task = task->next;
    (*task)(kn.knf_1x4, input_tensor, output_tensor, input_stride,
        output_stride, reg_alpha_full, reg_beta_full);

    task = task->next;
    (*task)(kn.knf_2x1, input_tensor, output_tensor, input_stride,
        output_stride, reg_alpha_full, reg_beta_full);

    task = task->next;
    (*task)(kn.knf_2x2, input_tensor, output_tensor, input_stride,
        output_stride, reg_alpha_full, reg_beta_full);

    task = task->next;
    (*task)(kn.knf_2x3, input_tensor, output_tensor, input_stride,
        output_stride, reg_alpha_full, reg_beta_full);

    task = task->next;
    (*task)(kn.knf_2x4, input_tensor, output_tensor, input_stride,
        output_stride, reg_alpha_full, reg_beta_full);

    task = task->next;
    (*task)(kn.knf_3x1, input_tensor, output_tensor, input_stride,
        output_stride, reg_alpha_full, reg_beta_full);

    task = task->next;
    (*task)(kn.knf_3x2, input_tensor, output_tensor, input_stride,
        output_stride, reg_alpha_full, reg_beta_full);

    task = task->next;
    (*task)(kn.knf_3x3, input_tensor, output_tensor, input_stride,
        output_stride, reg_alpha_full, reg_beta_full);

    task = task->next;
    (*task)(kn.knf_3x4, input_tensor, output_tensor, input_stride,
        output_stride, reg_alpha_full, reg_beta_full);

    task = task->next;
    (*task)(kn.knf_4x1, input_tensor, output_tensor, input_stride,
        output_stride, reg_alpha_full, reg_beta_full);

    task = task->next;
    (*task)(kn.knf_4x2, input_tensor, output_tensor, input_stride,
        output_stride, reg_alpha_full, reg_beta_full);

    task = task->next;
    (*task)(kn.knf_4x3, input_tensor, output_tensor, input_stride,
        output_stride, reg_alpha_full, reg_beta_full);

    task = task->next;
    (*task)(kn.knf_4x4, input_tensor, output_tensor, input_stride,
        output_stride, reg_alpha_full, reg_beta_full);

    task = task->next;
    (*task)(kn.knh_1x1, input_tensor, output_tensor, input_stride,
        output_stride, reg_alpha_half, reg_beta_half);

    task = task->next;
    (*task)(kn.knh_1x2, input_tensor, output_tensor, input_stride,
        output_stride, reg_alpha_half, reg_beta_half);

    task = task->next;
    (*task)(kn.knh_1x3, input_tensor, output_tensor, input_stride,
        output_stride, reg_alpha_half, reg_beta_half);

    task = task->next;
    (*task)(kn.knh_1x4, input_tensor, output_tensor, input_stride,
        output_stride, reg_alpha_half, reg_beta_half);

    task = task->next;
    (*task)(kn.knh_2x1, input_tensor, output_tensor, input_stride,
        output_stride, reg_alpha_half, reg_beta_half);

    task = task->next;
    (*task)(kn.knh_3x1, input_tensor, output_tensor, input_stride,
        output_stride, reg_alpha_half, reg_beta_half);

    task = task->next;
    (*task)(kn.knh_4x1, input_tensor, output_tensor, input_stride,
        output_stride, reg_alpha_half, reg_beta_half);

    task = task->next;
    (*task)(kn.kn_lin, input_tensor, output_tensor, 1, 0,
        reg_alpha_linear, reg_beta_linear);

    task = task->next;
    (*task)(kn.kn_lin, input_tensor, output_tensor, 1, 0,
        reg_alpha_linear, reg_beta_linear);
  }
}


template <typename ParamType>
INLINE void CGraphTrans<ParamType>::exec_common_leading_() {
  const auto ld_len = static_cast<TensorIdx>(this->param_->get_leading().first);

#pragma omp parallel for schedule(static)
  for (auto th_idx = 0; th_idx < this->threads_; ++th_idx)
    this->operations_[th_idx](this->param_->kn.kn_lin,
        this->param_->input_tensor, this->param_->output_tensor,
        ld_len, 0, this->param_->reg_alpha_linear,
        this->param_->reg_beta_linear);
}


/*
 * Import explicit instantiation declaration for class CGraphTrans, this file
 * should be generated by cmake script.
 */
#include "cgraph_trans_gen.tcc"

#endif // HPTC_CGRAPH_CGRAPH_TRANS_TCC_
