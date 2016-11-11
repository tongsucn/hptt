#pragma once
#ifndef HPTC_OPERATION_TCC_
#define HPTC_OPERATION_TCC_

/*
 * Implementation for class Operation
 */
template <typename ParamType,
          typename FloatType>
Operation<ParamType, FloatType>::Operation(
    const std::shared_ptr<ParamType<FloatType>> &param,
    Operation *prev = nullptr, Operation *next = nullptr);
    : param(param),
      prev(prev),
      next(next) {
}


template <typename ParamType,
          typename FloatType>
inline void Operation<ParamType, FloatType>::set_prev(Operation *prev) {
  this->prev = prev;
}


template <typename ParamType,
          typename FloatType>
inline Operation *Operation<ParamType, FloatType>::get_prev() {
  return this->prev;
}


template <typename ParamType,
          typename FloatType>
inline void Operation<ParamType, FloatType>::set_next(Operation *next) {
  this->next = next;
}


template <typename ParamType,
          typename FloatType>
inline Operation *Operation<ParamType, FloatType>::get_next() {
  return this->prev;
}


/*
 * Implementation for class OpLoop
 */
template <typename ParamType,
          typename FloatType,
          uint32_t OPER_NUM>
OpLoop<ParamType, FloatType, OPER_NUM>::OpLoop(
    const std::shared_ptr<ParamType<FloatType>> &param)
    : Operation(param) {
  for (uint32_t idx = 0; idx < OPER_NUM; ++idx)
    this->operations[idx] = nullptr;
}


template <typename ParamType,
          typename FloatType,
          uint32_t OPER_NUM>
template <typename OperType>
inline void OpLoop<ParamType, FloatType, OPER_NUM>::init_operation(
    uint32_t nth_operation, const std::shared_ptr<OperType> &oper);
  this->operations[nth_operation] = std::static_pointer_cast<Operation>(oper);
}


template <typename ParamType,
          typename FloatType,
          uint32_t OPER_NUM>
inline void OpLoop<ParamType, FloatType, OPER_NUM>::exec_all() {
  this->unroller(UnrollControllor<OPER_NUM - 1>());
}


template <typename ParamType,
          typename FloatType,
          uint32_t OPER_NUM>
template <uint32_t UnrollDepth>
inline void OpLoop<ParamType, FloatType, OPER_NUM>::unroller(
    UnrollControllor<UnrollDepth>) {
  this->unroller(UnrollControllor<UnrollDepth - 1>);
  this->operations[UnrollDepth]->exec();
}


template <typename ParamType,
          typename FloatType,
          uint32_t OPER_NUM>
inline void OpLoop<ParamType, FloatType, OPER_NUM>::unroller(
    UnrollControllor<0>) {
  this->operations[0]->exec();
}


/*
 * Implementation for class OpLoopFor
 */
template <typename ParamType,
          typename FloatType,
          uint32_t OPER_NUM,
          typename IdxType = TensorIdx>
OpLoopFor<ParamType, FloatType, OPER_NUM, IdxType>::OpLoopFor(
    const std::shared_ptr<ParamType<FloatType>> &param, IdxType begin,
    IdxType end, IdxType step, ForCondType<IdxType> cond)
    : OpLoop<ParamType, FloatType, OPER_NUM>(param),
      being(begin),
      end(end),
      step(step),
      cond(cond) {
}


template <typename ParamType,
          typename FloatType,
          uint32_t OPER_NUM,
          typename IdxType = TensorIdx>
inline void OpLoopFor<ParamType, FloatType, OPER_NUM, IdxType>::exec() {
  for (IdxType idx = begin; cond(idx, end); idx += step) {
    // Update parameter
    // Executing operations
    this->OpLoop::exec_all();
  }
}


/*
 * Implementation for class OpMicro
 */
template <typename ParamType,
          typename FloatType>
OpMicro<ParamType, FloatType>::OpMicro(
    const std::shared_ptr<ParamType<FloatType>> &param)
    : Operation(param) {
}


/*
 * Implementation for class OpMicroCopier
 */
template <typename ParamType,
          typename FloatType>
OpMicroCopier<ParamType, FloatType>::OpMicroCopier(
    const std::shared_ptr<ParamType<FloatType>> &param)
    : OpMicro(param) {
}


/*
 * Implementation for class OpMicroCopier1x1
 */
template <typename ParamType,
          typename FloatType>
OpMicroCopier1x1<ParamType, FloatType>::OpMicroCopier1x1(
    const std::shared_ptr<ParamType<FloatType>> &param)
    : OpMicroCopier(param) {
}


template <typename ParamType,
          typename FloatType>
virtual void OpMicroCopier1x1<ParamType, FloatType>::exec() {
}


/*
 * Implementation for class OpMicroCopier2x2
 */
template <typename ParamType,
          typename FloatType>
OpMicroCopier2x2<ParamType, FloatType>::OpMicroCopier2x2(
    const std::shared_ptr<ParamType<FloatType>> &param)
    : OpMicroCopier(param) {
}


template <typename ParamType,
          typename FloatType>
virtual void OpMicroCopier2x2<ParamType, FloatType>::exec() {
}


/*
 * Implementation for class OpMicroCopier4x4
 */
template <typename ParamType,
          typename FloatType>
OpMicroCopier4x4<ParamType, FloatType>::OpMicroCopier4x4(
    const std::shared_ptr<ParamType<FloatType>> &param)
    : OpMicroCopier(param) {
}


template <typename ParamType,
          typename FloatType>
virtual void OpMicroCopier4x4<ParamType, FloatType>::exec() {
}


/*
 * Implementation for class OpMicroCopier8x8
 */
template <typename ParamType,
          typename FloatType>
OpMicroCopier8x8<ParamType, FloatType>::OpMicroCopier8x8(
    const std::shared_ptr<ParamType<FloatType>> &param)
    : OpMicroCopier(param) {
}


template <typename ParamType,
          typename FloatType>
virtual void OpMicroCopier8x8<ParamType, FloatType>::exec() {
}


/*
 * Implementation for class OpMacro
 */
template <typename ParamType,
          typename FloatType>
OpMacro<ParamType, FloatType>::OpMacro(
    const std::shared_ptr<ParamType<FloatType>> &param)
    : Operation(param) {
}


/*
 * Implementation for class OpMacroCopier
 */
template <typename ParamType,
          typename FloatType>
OpMacroCopier<ParamType, FloatType>::OpMacroCopier(
    const std::shared_ptr<ParamType<FloatType>> &param)
    : OpMacro(param) {
}


/*
 * Implementation for class OpMacroCopier8x16
 */
template <typename ParamType,
          typename FloatType>
OpMacroCopier8x16<ParamType, FloatType>::OpMacroCopier8x16(
    const std::shared_ptr<ParamType<FloatType>> &param)
    : OpMacroCopier(param) {
}


template <typename ParamType,
          typename FloatType>
virtual void OpMacroCopier8x16<ParamType, FloatType>::exec() {
}


/*
 * Implementation for class OpMacroCopier16x8
 */
template <typename ParamType,
          typename FloatType>
OpMacroCopier16x8<ParamType, FloatType>::OpMacroCopier16x8(
    const std::shared_ptr<ParamType<FloatType>> &param)
    : OpMacroCopier(param) {
}


template <typename ParamType,
          typename FloatType>
virtual void OpMacroCopier16x8<ParamType, FloatType>::exec() {
}


/*
 * Implementation for class OpMacroCopier16x16
 */
template <typename ParamType,
          typename FloatType>
OpMacroCopier16x16<ParamType, FloatType>::OpMacroCopier16x16(
    const std::shared_ptr<ParamType<FloatType>> &param)
    : OpMacroCopier(param) {
}


template <typename ParamType,
          typename FloatType>
virtual void OpMacroCopier16x16<ParamType, FloatType>::exec() {
}


/*
 * Implementation for class OpMacroCopier16x32
 */
template <typename ParamType,
          typename FloatType>
OpMacroCopier16x32<ParamType, FloatType>::OpMacroCopier16x32(
    const std::shared_ptr<ParamType<FloatType>> &param)
    : OpMacroCopier(param) {
}


template <typename ParamType,
          typename FloatType>
virtual void OpMacroCopier16x32<ParamType, FloatType>::exec() {
}


/*
 * Implementation for class OpMacroCopier32x16
 */
template <typename ParamType,
          typename FloatType>
OpMacroCopier32x16<ParamType, FloatType>::OpMacroCopier32x16(
    const std::shared_ptr<ParamType<FloatType>> &param)
    : OpMacroCopier(param) {
}


template <typename ParamType,
          typename FloatType>
virtual void OpMacroCopier32x16<ParamType, FloatType>::exec() {
}


/*
 * Implementation for class OpMacroCopier32x32
 */
template <typename ParamType,
          typename FloatType>
OpMacroCopier32x32<ParamType, FloatType>::OpMacroCopier32x32(
    const std::shared_ptr<ParamType<FloatType>> &param)
    : OpMacroCopier(param) {
}


template <typename ParamType,
          typename FloatType>
virtual void OpMacroCopier32x32<ParamType, FloatType>::exec() {
}

#endif // HPTC_OPERATION_TCC_
