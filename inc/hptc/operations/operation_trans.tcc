#pragma once
#ifndef HPTC_OPERATIONS_OPERATION_TRANS_TCC_
#define HPTC_OPERATIONS_OPERATION_TRANS_TCC_

/*
 * Implementation for class OpForTransData
 */
template <TensorOrder ORDER,
          typename ParamType>
OpForTransData<ORDER, ParamType>::OpForTransData(
    std::shared_ptr<ParamType> &param) : param_(param) {
  // Initialize loop variables
  std::fill(this->loop_idx_, this->loop_idx_ + ORDER, 0);
  std::copy(this->loop_idx_, this->loop_idx_ + ORDER, this->loop_begin_);
  std::fill(this->loop_step_, this->loop_step_ + ORDER, 1);

  // Initialize loop indices
  for (TensorOrder idx = 0; idx < ORDER - param->merged_order; ++idx)
    this->loop_perm_idx_[idx] = &this->loop_idx_[idx];

  for (TensorOrder idx = ORDER - param->merged_order; idx < ORDER; ++idx)
    this->loop_perm_idx_[idx]
        = &this->loop_idx_[param->perm[idx] + ORDER - param->merged_order];
}


template <TensorOrder ORDER,
          typename ParamType>
INLINE void OpForTransData<ORDER, ParamType>::set_begin(
    TensorIdx begin_val, TensorIdx idx) {
  this->loop_begin_[idx] = begin_val;
}


template <TensorOrder ORDER,
          typename ParamType>
INLINE void OpForTransData<ORDER, ParamType>::set_end(
    TensorIdx end_val, TensorIdx idx) {
  this->loop_end_[idx] = end_val;
}


template <TensorOrder ORDER,
          typename ParamType>
INLINE void OpForTransData<ORDER, ParamType>::set_step(
    TensorIdx step_val, TensorIdx idx) {
  this->loop_step_[idx] = step_val;
}


/*
 * Specialization for class OpForTrans
 */
template <typename ParamType>
class OpForTrans<0, ParamType> final
    : public OpForTransData<0, ParamType> {
  OpForTrans() = delete;
};


template <typename ParamType>
class OpForTrans<1, ParamType> final
    : public OpForTransData<1, ParamType> {
  OpForTrans() = delete;
};


template <TensorOrder ORDER,
          typename ParamType>
OpForTrans<ORDER, ParamType>::OpForTrans(
    std::shared_ptr<ParamType> &param)
    : OpForTransData<ORDER, ParamType>(param),
      next(nullptr) {
}


template <TensorOrder ORDER,
          typename ParamType>
template <typename MacroType>
INLINE void OpForTrans<ORDER, ParamType>::operator()(MacroType &macro_kernel) {
  this->unroller(GenCounter<ORDER>(), macro_kernel);
}


template <TensorOrder ORDER,
          typename ParamType>
template <typename MacroType,
          GenNumType UNROLL_NUM>
INLINE void OpForTrans<ORDER, ParamType>::unroller(GenCounter<UNROLL_NUM>,
    MacroType &macro_kernel) {
  constexpr TensorOrder for_idx = ORDER - UNROLL_NUM;
  for (this->loop_idx_[for_idx] = this->loop_begin_[for_idx];
      this->loop_idx_[for_idx] < this->loop_end_[for_idx];
      this->loop_idx_[for_idx] += this->loop_step_[for_idx])
    this->unroller(GenCounter<UNROLL_NUM - 1>(), macro_kernel);
}


template <TensorOrder ORDER,
          typename ParamType>
template <typename MacroType>
INLINE void OpForTrans<ORDER, ParamType>::unroller(GenCounter<0>,
    MacroType &macro_kernel) {
  macro_kernel(&this->param_->input_tensor[this->loop_idx_],
      &this->param_->output_tensor[this->loop_perm_idx_],
      this->param_->input_stride, this->param_->output_stride);
}

#endif // HPTC_OPERATIONS_OPERATION_TRANS_TCC_
