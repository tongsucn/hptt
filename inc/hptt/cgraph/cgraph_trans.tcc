#pragma once
#ifndef HPTT_CGRAPH_CGRAPH_TRANS_TCC_
#define HPTT_CGRAPH_CGRAPH_TRANS_TCC_

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
HPTT_INL void CGraphTrans<ParamType>::exec() {
  if (this->thread_id_map_.size() == this->threads_) {
    auto th_idx = this->thread_id_map_[omp_get_thread_num()];
    if (not this->param_->is_common_leading())
      this->exec_general_(th_idx);
    else
      this->exec_common_leading_(th_idx);
  }
  else {
    if (not this->param_->is_common_leading()) {
#pragma omp parallel for schedule(static)
      for (TensorUInt th_idx = 0; th_idx < this->threads_; ++th_idx)
        this->exec_general_(th_idx);
    }
    else {
#pragma omp parallel for schedule(static)
      for (TensorUInt th_idx = 0; th_idx < this->threads_; ++th_idx)
        this->exec_common_leading_(th_idx);
    }
  }
}


template <typename ParamType>
HPTT_INL void CGraphTrans<ParamType>::operator()() {
  this->exec();
}


template <typename ParamType>
HPTT_INL typename CGraphTrans<ParamType>::Descriptor
CGraphTrans<ParamType>::get_descriptor() const {
  return this->descriptor_;
}


template <typename ParamType>
HPTT_INL void CGraphTrans<ParamType>::reset_data(const Float *data_in,
    Float *data_out) {
  this->param_->reset_data(data_in, data_out);
}


template <typename ParamType>
HPTT_INL void CGraphTrans<ParamType>::set_thread_ids(
    const std::vector<TensorInt> &thread_ids) {
  this->unset_thread_ids();
  if (thread_ids.size() == this->threads_)
    for (TensorUInt th_idx = 0; th_idx < this->threads_; ++th_idx)
      this->thread_id_map_[thread_ids[th_idx]] = th_idx;

  // If the map's size does not equal to thread number, then input thread ID
  // list must contain duplicate IDs
  if (this->thread_id_map_.size() != this->threads_)
    this->unset_thread_ids();
}


template <typename ParamType>
HPTT_INL void CGraphTrans<ParamType>::unset_thread_ids() {
  this->thread_id_map_.clear();
}


template <typename ParamType>
CGraphTrans<ParamType>::CGraphTrans(const std::shared_ptr<ParamType> &param,
    const Descriptor &descriptor)
    : param_(param),
      threads_(0),
      descriptor_(),
      operations_(nullptr),
      thread_id_map_() {
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
HPTT_INL void CGraphTrans<ParamType>::exec_general_(const TensorUInt th_idx) {
  const auto &kn = this->param_->get_kernel();
  const auto &input_tensor = this->param_->input_tensor;
  auto &output_tensor = this->param_->output_tensor;
  const auto stride_in_outld = this->param_->stride_in_outld;
  const auto stride_out_inld = this->param_->stride_out_inld;

  auto task = this->operations_ + th_idx;
  task->exec(kn.knf_1x1, input_tensor, output_tensor, stride_in_outld,
      stride_out_inld);

  task = task->next;
  task->exec(kn.knf_1x2, input_tensor, output_tensor, stride_in_outld,
      stride_out_inld);

  task = task->next;
  task->exec(kn.knf_1x3, input_tensor, output_tensor, stride_in_outld,
      stride_out_inld);

  task = task->next;
  task->exec(kn.knf_1x4, input_tensor, output_tensor, stride_in_outld,
      stride_out_inld);

  task = task->next;
  task->exec(kn.knf_2x1, input_tensor, output_tensor, stride_in_outld,
      stride_out_inld);

  task = task->next;
  task->exec(kn.knf_2x2, input_tensor, output_tensor, stride_in_outld,
      stride_out_inld);

  task = task->next;
  task->exec(kn.knf_2x3, input_tensor, output_tensor, stride_in_outld,
      stride_out_inld);

  task = task->next;
  task->exec(kn.knf_2x4, input_tensor, output_tensor, stride_in_outld,
      stride_out_inld);

  task = task->next;
  task->exec(kn.knf_3x1, input_tensor, output_tensor, stride_in_outld,
      stride_out_inld);

  task = task->next;
  task->exec(kn.knf_3x2, input_tensor, output_tensor, stride_in_outld,
      stride_out_inld);

  task = task->next;
  task->exec(kn.knf_3x3, input_tensor, output_tensor, stride_in_outld,
      stride_out_inld);

  task = task->next;
  task->exec(kn.knf_3x4, input_tensor, output_tensor, stride_in_outld,
      stride_out_inld);

  task = task->next;
  task->exec(kn.knf_4x1, input_tensor, output_tensor, stride_in_outld,
      stride_out_inld);

  task = task->next;
  task->exec(kn.knf_4x2, input_tensor, output_tensor, stride_in_outld,
      stride_out_inld);

  task = task->next;
  task->exec(kn.knf_4x3, input_tensor, output_tensor, stride_in_outld,
      stride_out_inld);

  task = task->next;
  task->exec(kn.knf_4x4, input_tensor, output_tensor, stride_in_outld,
      stride_out_inld);

  task = task->next;
  task->exec(kn.knh_1x1, input_tensor, output_tensor, stride_in_outld,
      stride_out_inld);

  task = task->next;
  task->exec(kn.knh_1x2, input_tensor, output_tensor, stride_in_outld,
      stride_out_inld);

  task = task->next;
  task->exec(kn.knh_1x3, input_tensor, output_tensor, stride_in_outld,
      stride_out_inld);

  task = task->next;
  task->exec(kn.knh_1x4, input_tensor, output_tensor, stride_in_outld,
      stride_out_inld);

  task = task->next;
  task->exec(kn.knh_2x1, input_tensor, output_tensor, stride_in_outld,
      stride_out_inld);

  task = task->next;
  task->exec(kn.knh_3x1, input_tensor, output_tensor, stride_in_outld,
      stride_out_inld);

  task = task->next;
  task->exec(kn.knh_4x1, input_tensor, output_tensor, stride_in_outld,
      stride_out_inld);

  task = task->next;
  task->exec(kn.kn_sca_right, input_tensor, output_tensor, 0, 0);

  task = task->next;
  task->exec(kn.kn_sca_bottom, input_tensor, output_tensor, 0, 0);

  task = task->next;
  task->exec(kn.kn_sca_scalar, input_tensor, output_tensor, 0, 0);
}


template <typename ParamType>
HPTT_INL void CGraphTrans<ParamType>::exec_common_leading_(
    const TensorUInt th_idx) {
  const auto &kn_core = this->param_->get_kernel().kn_lin_core;
  const auto &kn_right = this->param_->get_kernel().kn_lin_right;
  const auto &kn_bottom = this->param_->get_kernel().kn_lin_bottom;
  const auto &kn_scalar = this->param_->get_kernel().kn_lin_scalar;
  const auto &input_tensor = this->param_->input_tensor;
  auto &output_tensor = this->param_->output_tensor;
  const auto size_in_inld = this->param_->input_tensor.get_size(
      this->param_->begin_order_idx);
  const auto size_out_outld = this->param_->output_tensor.get_size(
      this->param_->begin_order_idx);

  auto task = this->operations_ + th_idx;
  task->exec(kn_core, input_tensor, output_tensor, size_in_inld,
      size_out_outld);

  task = task->next;
  task->exec(kn_right, input_tensor, output_tensor, size_in_inld,
      size_out_outld);

  task = task->next;
  task->exec(kn_bottom, input_tensor, output_tensor, size_in_inld,
      size_out_outld);

  task = task->next;
  task->exec(kn_scalar, input_tensor, output_tensor, size_in_inld,
      size_out_outld);
}


/*
 * Import explicit instantiation declaration for class CGraphTrans, this file
 * should be generated by cmake script.
 */
#include <hptt/gen/cgraph_trans_gen.tcc>

#endif // HPTT_CGRAPH_CGRAPH_TRANS_TCC_
