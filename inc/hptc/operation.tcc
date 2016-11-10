#pragma once
#ifndef HPTC_OPERATION_TCC_
#define HPTC_OPERATION_TCC_

/*
 * Implementation for class OpLoop
 */
template <uint32_t OPER_NUM>
OpLoop<OPER_NUM>::OpLoop(const std::shared_ptr<Param> &param)
    : Operation(param) {
  for (uint32_t idx = 0; idx < OPER_NUM; ++idx)
    this->operations[idx] = nullptr;
}


template <uint32_t OPER_NUM>
template <typename OperType>
inline void OpLoop<OPER_NUM>::init_operation(uint32_t nth_operation,
    const std::shared_ptr<OperType> &oper);
  this->operations[nth_operation] = std::static_pointer_cast<Operation>(oper);
}


template <uint32_t OPER_NUM>
inline void OpLoop<OPER_NUM>::exec_all() {
  this->unroller(UnrollControllor<OPER_NUM - 1>());
}


template <uint32_t OPER_NUM>
template <uint32_t UnrollDepth>
inline void OpLoop<OPER_NUM>::unroller(
    UnrollControllor<UnrollDepth>) {
  this->unroller(UnrollControllor<UnrollDepth - 1>);
  this->operations[UnrollDepth]->exec();
}


template <uint32_t OPER_NUM>
inline void OpLoop<OPER_NUM>::unroller(UnrollControllor<0>) {
  this->operations[0]->exec();
}


/*
 * Implementation for class OpLoopFor
 */
template <typename ParamType,
          uint32_t OPER_NUM,
          typename IdxType = TensorIdx>
OpLoopFor<ParamType, OPER_NUM, IdxType>::OpLoopFor(
    const std::shared_ptr<ParamType> &param, IdxType begin, IdxType end,
    IdxType step, ForCondType<IdxType> cond)
    : OpLoop<OPER_NUM>(std::static_pointer_cast<Param>(param)),
      being(begin),
      end(end),
      step(step),
      cond(cond) {
}


template <typename ParamType,
          uint32_t OPER_NUM,
          typename IdxType = TensorIdx>
inline void OpLoopFor<ParamType, OPER_NUM, IdxType>::exec() {
  for (IdxType idx = begin; cond(idx, end); idx += step) {
    // Update parameter
    // Executing operations
    this->OpLoop::exec_all();
  }
}


/*
 * Implementation for class OpMicroCopier1x1
 */
template <typename ParamType>
OpMicroCopier1x1<ParamType>::OpMicroCopier1x1(
    const std::shared_ptr<ParamType> &param)
    : OpMicroCopier(param) {
}


template <typename ParamType>
virtual void OpMicroCopier1x1<ParamType>::exec() {
}


/*
 * Implementation for class OpMicroCopier2x2
 */
template <typename ParamType>
OpMicroCopier2x2<ParamType>::OpMicroCopier2x2(
    const std::shared_ptr<ParamType> &param)
    : OpMicroCopier(param) {
}


template <typename ParamType>
virtual void OpMicroCopier2x2<ParamType>::exec() {
}


/*
 * Implementation for class OpMicroCopier4x4
 */
template <typename ParamType>
OpMicroCopier4x4<ParamType>::OpMicroCopier4x4(
    const std::shared_ptr<ParamType> &param)
    : OpMicroCopier(param) {
}


template <typename ParamType>
virtual void OpMicroCopier4x4<ParamType>::exec() {
}


/*
 * Implementation for class OpMicroCopier8x8
 */
template <typename ParamType>
OpMicroCopier8x8<ParamType>::OpMicroCopier8x8(
    const std::shared_ptr<ParamType> &param)
    : OpMicroCopier(param) {
}


template <typename ParamType>
virtual void OpMicroCopier8x8<ParamType>::exec() {
}


/*
 * Implementation for class OpMacroCopier8x16
 */
template <typename ParamType>
OpMacroCopier8x16<ParamType>::OpMacroCopier8x16(
    const std::shared_ptr<ParamType> &param)
    : OpMacroCopier(param) {
}


template <typename ParamType>
virtual void OpMacroCopier8x16<ParamType>::exec() {
}


/*
 * Implementation for class OpMacroCopier16x8
 */
template <typename ParamType>
OpMacroCopier16x8<ParamType>::OpMacroCopier16x8(
    const std::shared_ptr<ParamType> &param)
    : OpMacroCopier(param) {
}


template <typename ParamType>
virtual void OpMacroCopier16x8<ParamType>::exec() {
}


/*
 * Implementation for class OpMacroCopier16x16
 */
template <typename ParamType>
OpMacroCopier16x16<ParamType>::OpMacroCopier16x16(
    const std::shared_ptr<ParamType> &param)
    : OpMacroCopier(param) {
}


template <typename ParamType>
virtual void OpMacroCopier16x16<ParamType>::exec() {
}


/*
 * Implementation for class OpMacroCopier16x32
 */
template <typename ParamType>
OpMacroCopier16x32<ParamType>::OpMacroCopier16x32(
    const std::shared_ptr<ParamType> &param)
    : OpMacroCopier(param) {
}


template <typename ParamType>
virtual void OpMacroCopier16x32<ParamType>::exec() {
}


/*
 * Implementation for class OpMacroCopier32x16
 */
template <typename ParamType>
OpMacroCopier32x16<ParamType>::OpMacroCopier32x16(
    const std::shared_ptr<ParamType> &param)
    : OpMacroCopier(param) {
}


template <typename ParamType>
virtual void OpMacroCopier32x16<ParamType>::exec() {
}


/*
 * Implementation for class OpMacroCopier32x32
 */
template <typename ParamType>
OpMacroCopier32x32<ParamType>::OpMacroCopier32x32(
    const std::shared_ptr<ParamType> &param)
    : OpMacroCopier(param) {
}


template <typename ParamType>
virtual void OpMacroCopier32x32<ParamType>::exec() {
}

#endif // HPTC_OPERATION_TCC_
