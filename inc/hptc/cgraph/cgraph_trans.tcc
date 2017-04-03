#pragma once
#ifndef HPTC_CGRAPH_CGRAPH_TRANS_TCC_
#define HPTC_CGRAPH_CGRAPH_TRANS_TCC_

/*
 * Implementation for class CGraphTrans::Descriptor
 */
template <typename ParamType>
CGraphTrans<ParamType>::Descriptor::Descriptor()
    : description(1) {
  for (TensorUInt order_idx = 0; order_idx < ORDER; ++order_idx)
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
HPTC_INL void CGraphTrans<ParamType>::exec() {
  if (not this->param_->is_common_leading())
    this->exec_general_();
  else
    this->exec_common_leading_();
}


template <typename ParamType>
HPTC_INL void CGraphTrans<ParamType>::operator()() {
  this->exec();
}


template <typename ParamType>
HPTC_INL typename CGraphTrans<ParamType>::Descriptor
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
  this->operations_ = new OpForTrans<ORDER> [this->threads_];

  // Initialize for loops' parameters and loop order
  // Set kernel index's end, leave two for scalar kernels
  auto &description = this->descriptor_.description;
  const auto kn_idx_end = description[0].size();
  for (TensorUInt th_idx = 0; th_idx < this->threads_; ++th_idx) {
    auto curr_oper = this->operations_ + th_idx;
    curr_oper->init(this->descriptor_.loop_order, description[th_idx][0],
        this->param_->begin_order_idx, this->param_->perm);

    for (TensorUInt kn_idx = 1; kn_idx < kn_idx_end; ++kn_idx) {
      curr_oper->next = new OpForTrans<ORDER>(this->descriptor_.loop_order,
          description[th_idx][kn_idx], this->param_->begin_order_idx,
          this->param_->perm);
      curr_oper = curr_oper->next;
    }
  }
}


template <typename ParamType>
void CGraphTrans<ParamType>::release_() {
  // Release operations
  for (decltype(this->threads_) th_idx = 0; th_idx < this->threads_; ++th_idx) {
    auto curr_oper = this->operations_[th_idx].next;
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
HPTC_INL void CGraphTrans<ParamType>::exec_general_() {
  const auto &kn = this->param_->get_kernel();
  const auto &input_tensor = this->param_->input_tensor;
  auto &output_tensor = this->param_->output_tensor;
  const auto input_stride = this->param_->input_stride;
  const auto output_stride = this->param_->output_stride;

#pragma omp parallel for schedule(static)
  for (decltype(this->threads_) th_idx = 0; th_idx < this->threads_; ++th_idx) {
    auto task = this->operations_ + th_idx;
    task->exec(kn.knf_1x1, input_tensor, output_tensor, input_stride,
        output_stride);

    task = task->next;
    task->exec(kn.knf_1x2, input_tensor, output_tensor, input_stride,
        output_stride);

    task = task->next;
    task->exec(kn.knf_1x3, input_tensor, output_tensor, input_stride,
        output_stride);

    task = task->next;
    task->exec(kn.knf_1x4, input_tensor, output_tensor, input_stride,
        output_stride);

    task = task->next;
    task->exec(kn.knf_2x1, input_tensor, output_tensor, input_stride,
        output_stride);

    task = task->next;
    task->exec(kn.knf_2x2, input_tensor, output_tensor, input_stride,
        output_stride);

    task = task->next;
    task->exec(kn.knf_2x3, input_tensor, output_tensor, input_stride,
        output_stride);

    task = task->next;
    task->exec(kn.knf_2x4, input_tensor, output_tensor, input_stride,
        output_stride);

    task = task->next;
    task->exec(kn.knf_3x1, input_tensor, output_tensor, input_stride,
        output_stride);

    task = task->next;
    task->exec(kn.knf_3x2, input_tensor, output_tensor, input_stride,
        output_stride);

    task = task->next;
    task->exec(kn.knf_3x3, input_tensor, output_tensor, input_stride,
        output_stride);

    task = task->next;
    task->exec(kn.knf_3x4, input_tensor, output_tensor, input_stride,
        output_stride);

    task = task->next;
    task->exec(kn.knf_4x1, input_tensor, output_tensor, input_stride,
        output_stride);

    task = task->next;
    task->exec(kn.knf_4x2, input_tensor, output_tensor, input_stride,
        output_stride);

    task = task->next;
    task->exec(kn.knf_4x3, input_tensor, output_tensor, input_stride,
        output_stride);

    task = task->next;
    task->exec(kn.knf_4x4, input_tensor, output_tensor, input_stride,
        output_stride);

    task = task->next;
    task->exec(kn.knh_1x1, input_tensor, output_tensor, input_stride,
        output_stride);

    task = task->next;
    task->exec(kn.knh_1x2, input_tensor, output_tensor, input_stride,
        output_stride);

    task = task->next;
    task->exec(kn.knh_1x3, input_tensor, output_tensor, input_stride,
        output_stride);

    task = task->next;
    task->exec(kn.knh_1x4, input_tensor, output_tensor, input_stride,
        output_stride);

    task = task->next;
    task->exec(kn.knh_2x1, input_tensor, output_tensor, input_stride,
        output_stride);

    task = task->next;
    task->exec(kn.knh_3x1, input_tensor, output_tensor, input_stride,
        output_stride);

    task = task->next;
    task->exec(kn.knh_4x1, input_tensor, output_tensor, input_stride,
        output_stride);

    task = task->next;
    task->exec(kn.kn_scl, input_tensor, output_tensor, 1, 0);

    task = task->next;
    task->exec(kn.kn_scl, input_tensor, output_tensor, 1, 0);
  }
}


template <typename ParamType>
HPTC_INL void CGraphTrans<ParamType>::exec_common_leading_() {
  const auto &kn_core = this->param_->get_kernel().kn_lin_core;
  const auto &kn_right = this->param_->get_kernel().kn_lin_right;
  const auto &kn_bottom = this->param_->get_kernel().kn_lin_bottom;
  const auto &kn_scalar = this->param_->get_kernel().kn_lin_scalar;
  const auto &input_tensor = this->param_->input_tensor;
  auto &output_tensor = this->param_->output_tensor;
  const auto ld_in_len = this->param_->input_tensor.get_size(
      this->param_->begin_order_idx);
  const auto ld_out_len = this->param_->output_tensor.get_size(
      this->param_->begin_order_idx);

#pragma omp parallel for schedule(static)
  for (decltype(this->threads_) th_idx = 0; th_idx < this->threads_; ++th_idx) {
    auto task = this->operations_ + th_idx;
    task->exec(kn_core, input_tensor, output_tensor, ld_in_len, ld_out_len);

    task = task->next;
    task->exec(kn_right, input_tensor, output_tensor, ld_in_len, ld_out_len);

    task = task->next;
    task->exec(kn_bottom, input_tensor, output_tensor, ld_in_len, ld_out_len);

    task = task->next;
    task->exec(kn_scalar, input_tensor, output_tensor, ld_in_len, ld_out_len);
  }
}


/*
 * Import explicit instantiation declaration for class CGraphTrans, this file
 * should be generated by cmake script.
 */
#include <hptc/gen/cgraph_trans_gen.tcc>

#endif // HPTC_CGRAPH_CGRAPH_TRANS_TCC_
