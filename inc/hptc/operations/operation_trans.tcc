#pragma once
#ifndef HPTC_OPERATIONS_OPERATION_TRANS_TCC_
#define HPTC_OPERATIONS_OPERATION_TRANS_TCC_

/*
 * Specialization for class OpForTrans
 */
template <typename ParamType>
class OpForTrans<ParamType, 0> {
  OpForTrans() = delete;
};


template <typename ParamType>
class OpForTrans<ParamType, 1> {
  OpForTrans() = delete;
};


/*
 * Implementation for class OpForTrans
 */
template <typename ParamType,
          TensorOrder ORDER>
OpForTrans<ParamType, ORDER>::OpForTrans()
    : next(nullptr),
      param_(nullptr) {
  this->init_disable_();
}


template <typename ParamType,
          TensorOrder ORDER>
OpForTrans<ParamType, ORDER>::OpForTrans(
    const std::shared_ptr<ParamType> &param, const LoopOrder<ORDER> &loop_order,
    const LoopParam<ORDER> &loops)
    : next(nullptr),
      param_(param) {
  this->init(param, loop_order, loops);
}


template <typename ParamType,
          TensorOrder ORDER>
void OpForTrans<ParamType, ORDER>::init(
    const std::shared_ptr<ParamType> &param, const LoopOrder<ORDER> &loop_order,
    const LoopParam<ORDER> &loops) {
  this->param_ = param;

  // Initialize loops
  this->init_loops_(loop_order, loops);

  // Initialize permutation array
  this->init_perm_idx_();
}


template <typename ParamType,
          TensorOrder ORDER>
template <typename MacroType>
INLINE void OpForTrans<ParamType, ORDER>::operator()(MacroType &macro_kernel) {
  this->unroller_(GenCounter<ORDER>(), macro_kernel);
}


template <typename ParamType,
          TensorOrder ORDER>
void OpForTrans<ParamType, ORDER>::init_disable_() {
  std::fill(this->loop_idx_, this->loop_idx_ + ORDER, 0);
  std::fill(this->loop_perm_idx_, this->loop_perm_idx_ + ORDER, nullptr);
  std::fill(this->loop_begin_, this->loop_begin_ + ORDER, 0);
  std::fill(this->loop_end_, this->loop_end_ + ORDER, 0);
  std::fill(this->loop_step_, this->loop_step_ + ORDER, 0);

  for (TensorOrder idx = 0; idx < ORDER; ++idx)
    this->loop_order_[idx] = idx;
}


template <typename ParamType,
          TensorOrder ORDER>
void OpForTrans<ParamType, ORDER>::init_loops_(
    const LoopOrder<ORDER> &loop_order,const LoopParam<ORDER> &loop) {
  // Initialize loop order
  std::copy(loop_order.begin(), loop_order.end(), this->loop_order_);

  // Initialize loops
  std::copy(loop.loop_begin, loop.loop_begin + ORDER, this->loop_begin_);
  std::copy(loop.loop_end, loop.loop_end + ORDER, this->loop_end_);
  std::copy(loop.loop_step, loop.loop_step + ORDER, this->loop_step_);
}


template <typename ParamType,
          TensorOrder ORDER>
INLINE void OpForTrans<ParamType, ORDER>::init_perm_idx_() {
  auto end_idx = ORDER - this->param_->merged_order;
  for (TensorOrder idx = 0; idx < end_idx; ++idx)
    this->loop_perm_idx_[idx] = &this->loop_idx_[idx];

  for (TensorOrder idx = end_idx; idx < ORDER; ++idx)
    this->loop_perm_idx_[idx] = &this->loop_idx_[this->param_->perm[idx] + ORDER
        - this->param_->merged_order];
}


template <typename ParamType,
          TensorOrder ORDER>
template <typename MacroType,
          GenNumType UNROLL_NUM>
INLINE void OpForTrans<ParamType, ORDER>::unroller_(GenCounter<UNROLL_NUM>,
    MacroType &macro_kernel) {
  auto for_idx = this->loop_order_[ORDER - UNROLL_NUM];
  for (this->loop_idx_[for_idx] = this->loop_begin_[for_idx];
      this->loop_idx_[for_idx] < this->loop_end_[for_idx];
      this->loop_idx_[for_idx] += this->loop_step_[for_idx])
    this->unroller_(GenCounter<UNROLL_NUM - 1>(), macro_kernel);
}


template <typename ParamType,
          TensorOrder ORDER>
template <typename MacroType>
INLINE void OpForTrans<ParamType, ORDER>::unroller_(GenCounter<0>,
    MacroType &macro_kernel) {
  macro_kernel(&this->param_->input_tensor[this->loop_idx_],
      &this->param_->output_tensor[this->loop_perm_idx_],
      this->param_->input_stride, this->param_->output_stride);
}

#endif // HPTC_OPERATIONS_OPERATION_TRANS_TCC_
