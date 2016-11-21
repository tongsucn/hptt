#pragma once
#ifndef HPTC_OPERATION_BASE_TCC_
#define HPTC_OPERATION_BASE_TCC_


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
inline void Operation<FloatType, ParamType>::set_prev(Operation *prev) {
  this->prev = prev;
}


template <typename FloatType,
          template <typename> typename ParamType>
inline Operation<FloatType, ParamType> *
Operation<FloatType, ParamType>::get_prev() {
  return this->prev;
}


template <typename FloatType,
          template <typename> typename ParamType>
inline void Operation<FloatType, ParamType>::set_next(Operation *next) {
  this->next = next;
}


template <typename FloatType,
          template <typename> typename ParamType>
inline Operation<FloatType, ParamType> *
Operation<FloatType, ParamType>::get_next() {
  return this->prev;
}


/*
 * Implementation for class OpLoop
 */
template <typename FloatType,
          template <typename> typename ParamType,
          uint32_t OPER_NUM>
OpLoop<FloatType, ParamType, OPER_NUM>::OpLoop(
    const std::shared_ptr<ParamType<FloatType>> &param)
    : Operation<FloatType, ParamType>(param) {
  for (uint32_t idx = 0; idx < OPER_NUM; ++idx)
    this->operations[idx] = nullptr;
}


template <typename FloatType,
          template <typename> typename ParamType,
          uint32_t OPER_NUM>
template <typename OperType>
inline void OpLoop<FloatType, ParamType, OPER_NUM>::init_operation(
    uint32_t operation_idx, const std::shared_ptr<OperType> &oper) {
  this->operations[operation_idx]
    = std::static_pointer_cast<Operation<FloatType, ParamType>>(oper);
}


template <typename FloatType,
          template <typename> typename ParamType,
          uint32_t OPER_NUM>
inline void OpLoop<FloatType, ParamType, OPER_NUM>::exec_all() {
  op_arr_unroller(this->operations, UnrollControllor<OPER_NUM - 1>());
}


/*
 * Implementation for class OpLoopFor
 */
template <typename FloatType,
          template <typename> typename ParamType,
          uint32_t OPER_NUM>
OpLoopFor<FloatType, ParamType, OPER_NUM>::OpLoopFor(
    const std::shared_ptr<ParamType<FloatType>> &param, TensorIdx &target_idx,
    TensorIdx begin, TensorIdx end, TensorIdx step,
    ForCondType cond = ForCond::Smaller)
    : OpLoop<FloatType, ParamType, OPER_NUM>(param),
      curr_idx(target_idx),
      begin(begin),
      end(end),
      step(step),
      cond(cond) {
}


template <typename FloatType,
          template <typename> typename ParamType,
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
          template <typename> typename ParamType,
          uint32_t HEIGHT,
          uint32_t WIDTH>
OpMicro<FloatType, ParamType, HEIGHT, WIDTH>::OpMicro(
    const std::shared_ptr<ParamType<FloatType>> &param)
    : Operation<FloatType, ParamType>(param) {
}


/*
 * Implementation for class OpMacro
 */
template <typename FloatType,
          template <typename> typename ParamType,
          uint32_t OPER_NUM>
OpMacro<FloatType, ParamType, OPER_NUM>::OpMacro(
    const std::shared_ptr<ParamType<FloatType>> &param)
    : Operation<FloatType, ParamType>(param) {
  for (TensorIdx idx = 0; idx < OPER_NUM; ++idx)
    this->operations[idx] = nullptr;
}


template <typename FloatType,
          template <typename> typename ParamType,
          uint32_t OPER_NUM>
template <typename UnrollerType, UnrollerType unroller>
inline void OpMacro<FloatType, ParamType, OPER_NUM>::exec_all() {
  unroller(this->operations, UnrollControllor<OPER_NUM - 1>());
}


template <typename FloatType,
          template <typename> typename ParamType,
          uint32_t OPER_NUM>
inline void OpMacro<FloatType, ParamType, OPER_NUM>::init_operation(
    uint32_t operation_idx, Operation<FloatType, ParamType> *oper) {
  this->operations[operation_idx] = oper;
}

#endif // HPTC_OPERATION_BASE_TCC_
