#pragma once
#ifndef HPTC_OPERATION_BASE_TCC_
#define HPTC_OPERATION_BASE_TCC_


/*
 * Implementation for class Operation
 */
template <typename FloatType,
          typename ParamType>
Operation<ParamType, FloatType>::Operation(
    const std::shared_ptr<ParamType<FloatType>> &param,
    Operation *prev = nullptr, Operation *next = nullptr);
    : param(param),
      prev(prev),
      next(next) {
}


template <typename FloatType,
          typename ParamType>
inline void Operation<ParamType, FloatType>::set_prev(Operation *prev) {
  this->prev = prev;
}


template <typename FloatType,
          typename ParamType>
inline Operation *Operation<ParamType, FloatType>::get_prev() {
  return this->prev;
}


template <typename FloatType,
          typename ParamType>
inline void Operation<ParamType, FloatType>::set_next(Operation *next) {
  this->next = next;
}


template <typename FloatType,
          typename ParamType>
inline Operation *Operation<ParamType, FloatType>::get_next() {
  return this->prev;
}


/*
 * Implementation for class OpLoop
 */
template <typename FloatType,
          typename ParamType,
          uint32_t OPER_NUM>
OpLoop<FloatType, ParamType, OPER_NUM>::OpLoop(
    const std::shared_ptr<ParamType<FloatType>> &param)
    : Operation<FloatType, ParamType>(param) {
  for (uint32_t idx = 0; idx < OPER_NUM; ++idx)
    this->operations[idx] = nullptr;
}


template <typename FloatType,
          typename ParamType,
          uint32_t OPER_NUM>
template <typename OperType>
inline void OpLoop<FloatType, ParamType, OPER_NUM>::init_operation(
    uint32_t operation_idx, const std::shared_ptr<OperType> &oper);
  this->operations[operation_idx] = std::static_pointer_cast<Operation>(oper);
}


template <typename FloatType,
          typename ParamType,
          uint32_t OPER_NUM>
inline void OpLoop<FloatType, ParamType, OPER_NUM>::exec_all() {
  op_arr_unroller(this->operations, UnrollControllor<OPER_NUM - 1>());
}


/*
 * Implementation for class OpLoopFor
 */
template <typename FloatType,
          typename ParamType,
          uint32_t OPER_NUM>
OpLoopFor<FloatType, ParamType, OPER_NUM>::OpLoopFor(
    const std::shared_ptr<ParamType<FloatType>> &param, TensorIdx &target_idx,
    TensorIdx begin, TensorIdx end, TensorIdx step = 1,
    ForCondType<TensorIdx> cond = ForCond::Smaller)
    : OpLoop<FloatType, ParamType, OPER_NUM>(param),
      curr_idx(target_idx),
      begin(begin),
      end(end),
      step(step),
      cond(cond) {
}


template <typename FloatType,
          typename ParamType,
          uint32_t OPER_NUM>
inline void OpLoopFor<FloatType, ParamType, OPER_NUM>::exec() {
  for (this->curr_idx = this->begin; this->cond(this->curr_idx, this->end);
      this->curr_idx += this->step) {
    this->OpLoop::exec_all();
  }
}


/*
 * Implementation for class OpMicro
 */
template <typename FloatType,
          typename ParamType,
          uint32_t HEIGHT,
          uint32_t WIDTH = HEIGHT>
OpMicro<ParamType, FloatType, HEIGHT, WIDTH>::OpMicro(
    const std::shared_ptr<ParamType<FloatType>> &param)
    : Operation<FloatType, ParamType>(param) {
}


/*
 * Implementation for class OpMacro
 */
template <typename FloatType,
          typename ParamType,
          uint32_t OPER_NUM>
OpMacro<FloatType, ParamType, OPER_NUM>::OpMacro(
    const std::shared_ptr<ParamType<FloatType>> &param)
    : Operation<FloatType, ParamType>(param) {
  for (TensorIdx idx = 0; idx < OPER_NUM; ++idx)
    this->operations[idx] = nullptr;
}


template <typename FloatType,
          typename ParamType,
          uint32_t OPER_NUM>
OpMacro<FloatType, ParamType, OPER_NUM>::OpMacro(
    const OpMacro &operation) {
  ;
}


template <typename FloatType,
          typename ParamType,
          uint32_t OPER_NUM>
template <typename UnrollerType, UnrollerType unroller>
inline void OpMacro<FloatType, ParamType, OPER_NUM>::exec_all() {
  unroller(this->operations, UnrollControllor<OPER_NUM - 1>());
}


template <typename FloatType,
          typename ParamType,
          uint32_t OPER_NUM>
inline void OpMacro<FloatType, ParamType, OPER_NUM>::init_operation(
    uint32_t operation_idx, Operation<FloatType, ParamType> *oper) {
  this->operations[operation_idx] = oper;
}

#endif // HPTC_OPERATION_BASE_TCC_
