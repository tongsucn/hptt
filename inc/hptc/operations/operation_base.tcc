#pragma once
#ifndef HPTC_OPERATIONS_OPERATION_BASE_TCC_
#define HPTC_OPERATIONS_OPERATION_BASE_TCC_

/*
 * Implementation for class Operation
 */
template <typename FloatType,
          template <typename> typename ParamType>
Operation<FloatType, ParamType>::Operation(
    const std::shared_ptr<ParamType<FloatType>> &param,
    Operation *prev, Operation *next)
    : param(param),
      prev(prev),
      next(next) {
}


template <typename FloatType,
          template <typename> typename ParamType>
INLINE void Operation<FloatType, ParamType>::set_prev(Operation *prev) {
  this->prev = prev;
}


template <typename FloatType,
          template <typename> typename ParamType>
INLINE Operation<FloatType, ParamType> *
Operation<FloatType, ParamType>::get_prev() {
  return this->prev;
}


template <typename FloatType,
          template <typename> typename ParamType>
INLINE void Operation<FloatType, ParamType>::set_next(Operation *next) {
  this->next = next;
}


template <typename FloatType,
          template <typename> typename ParamType>
INLINE Operation<FloatType, ParamType> *
Operation<FloatType, ParamType>::get_next() {
  return this->prev;
}


/*
 * Implementation for class OpLoop
 */
template <typename FloatType,
          template <typename> typename ParamType,
          GenNumType OPER_NUM>
OpLoop<FloatType, ParamType, OPER_NUM>::OpLoop(
    const std::shared_ptr<ParamType<FloatType>> &param)
    : Operation<FloatType, ParamType>(param) {
  for (GenNumType idx = 0; idx < OPER_NUM; ++idx)
    this->operations_[idx] = nullptr;
}


template <typename FloatType,
          template <typename> typename ParamType,
          GenNumType OPER_NUM>
template <typename OperType>
INLINE void OpLoop<FloatType, ParamType, OPER_NUM>::init_operation(
    const std::shared_ptr<OperType> &oper, GenNumType operation_idx) {
  this->operations_[operation_idx]
    = std::static_pointer_cast<Operation<FloatType, ParamType>>(oper);
}


template <typename FloatType,
          template <typename> typename ParamType,
          GenNumType OPER_NUM>
INLINE void OpLoop<FloatType, ParamType, OPER_NUM>::exec_all() {
  op_arr_unroller(this->operations_, GenCounter<OPER_NUM - 1>());
}


/*
 * Implementation for class OpLoopFor
 */
template <typename FloatType,
          template <typename> typename ParamType,
          GenNumType OPER_NUM>
OpLoopFor<FloatType, ParamType, OPER_NUM>::OpLoopFor(
    const std::shared_ptr<ParamType<FloatType>> &param, TensorIdx &target_idx,
    TensorIdx begin, TensorIdx end, TensorIdx step, ForCondType cond)
    : OpLoop<FloatType, ParamType, OPER_NUM>(param),
      curr_idx_(target_idx),
      begin_(begin),
      end_(end),
      step_(step),
      cond_(cond) {
}


template <typename FloatType,
          template <typename> typename ParamType,
          GenNumType OPER_NUM>
INLINE void OpLoopFor<FloatType, ParamType, OPER_NUM>::exec() {
  for (this->curr_idx_ = this->begin_; this->cond_(this->curr_idx_, this->end_);
      this->curr_idx_ += this->step_) {
    this->exec_all();
  }
}


/*
 * Implementation for class OpMacro
 */
template <typename FloatType,
          template <typename> typename ParamType>
OpMacro<FloatType, ParamType>::OpMacro(
    const std::shared_ptr<ParamType<FloatType>> &param)
    : Operation<FloatType, ParamType>(param) {
}

#endif // HPTC_OPERATIONS_OPERATION_BASE_TCC_
