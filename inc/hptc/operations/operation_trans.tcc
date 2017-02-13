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
    const std::shared_ptr<ParamType> &param)
    : next(nullptr),
      param_(param) {
  this->init_disable_();
  this->init_perm_idx_();
}


template <typename ParamType,
          TensorOrder ORDER>
OpForTrans<ParamType, ORDER>::OpForTrans(const OpForTrans &loop_data)
    : next(nullptr),
      param_(loop_data.param_) {
  std::copy(loop_data.loop_idx_, loop_data.loop_idx_ + ORDER, this->loop_idx_);
  this->init_perm_idx_();

  std::copy(loop_data.loop_begin_, loop_data.loop_begin_ + ORDER,
      this->loop_begin_);
  std::copy(loop_data.loop_end_, loop_data.loop_end_ + ORDER, this->loop_end_);
  std::copy(loop_data.loop_step_, loop_data.loop_step_ + ORDER,
      this->loop_step_);
  std::copy(loop_data.loop_order_, loop_data.loop_order_ + ORDER,
      this->loop_order_);
}


template <typename ParamType,
          TensorOrder ORDER>
OpForTrans<ParamType, ORDER> &OpForTrans<ParamType, ORDER>::operator=(
    const OpForTrans &loop_data) {
  // Do not copy next pointer
  this->param_ = loop_data.param_;
  std::copy(loop_data.loop_idx_, loop_data.loop_idx_ + ORDER, this->loop_idx_);
  this->init_perm_idx_();

  std::copy(loop_data.loop_begin_, loop_data.loop_begin_ + ORDER,
      this->loop_begin_);
  std::copy(loop_data.loop_end_, loop_data.loop_end_ + ORDER, this->loop_end_);
  std::copy(loop_data.loop_step_, loop_data.loop_step_ + ORDER,
      this->loop_step_);
  std::copy(loop_data.loop_order_, loop_data.loop_order_ + ORDER,
      this->loop_order_);

  return *this;
}


template <typename ParamType,
          TensorOrder ORDER>
void OpForTrans<ParamType, ORDER>::init(
    const std::shared_ptr<ParamType> &param) {
  this->param_ = param;
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
INLINE TensorIdx &OpForTrans<ParamType, ORDER>::begin(TensorIdx idx) {
  return this->loop_begin_[idx];
}


template <typename ParamType,
          TensorOrder ORDER>
INLINE TensorIdx &OpForTrans<ParamType, ORDER>::end(TensorIdx idx) {
  return this->loop_end_[idx];
}


template <typename ParamType,
          TensorOrder ORDER>
INLINE TensorIdx &OpForTrans<ParamType, ORDER>::step(TensorIdx idx) {
  return this->loop_step_[idx];
}


template <typename ParamType,
          TensorOrder ORDER>
INLINE void OpForTrans<ParamType, ORDER>::set_order(
    const std::array<TensorOrder, ORDER> &order) {
  std::copy(order.begin(), order.end(), this->loop_order_);
}


template <typename ParamType,
          TensorOrder ORDER>
INLINE void OpForTrans<ParamType, ORDER>::set_pass(TensorOrder order) {
  std::fill(this->loop_idx_, this->loop_idx_ + order, 0);
  std::fill(this->loop_end_, this->loop_end_ + order, 1);
  std::fill(this->loop_step_, this->loop_step_ + order, 1);
}


template <typename ParamType,
          TensorOrder ORDER>
INLINE const TensorIdx *OpForTrans<ParamType, ORDER>::get_order() const {
  return this->loop_order_;
}


template <typename ParamType,
          TensorOrder ORDER>
INLINE bool OpForTrans<ParamType, ORDER>::is_disable() {
  return this->loop_idx_[0] >= this->loop_end_[0];
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
