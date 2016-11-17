#pragma once
#ifndef HPTC_OPERATION_BASE_TCC_
#define HPTC_OPERATION_BASE_TCC_


/*
 * Implementation for class OpMicroTrans
 */
template <typename FloatType,
          typename ParamType,
          uint32_t HEIGHT,
          uint32_t WIDTH>
OpMicroTrans<FloatType, ParamType, HEIGHT, WIDTH>::OpMicroTrans(
    const std::shared_ptr<ParamType<FloatType>> &param)
    : OpMicro<FloatType, ParamType, HEIGHT, WIDTH>(param) {
}


template <typename FloatType,
          typename ParamType,
          uint32_t HEIGHT,
          uint32_t WIDTH = HEIGHT>
void OpMicroTrans<FloatType, ParamType, HEIGHT, WIDTH>::exec() {
}


/*
 * Implementation for class OpMacroTrans
 */
template <typename FloatType,
          typename MicroType,
          uint32_t HEIGHT,
          uint32_t WIDTH = HEIGHT>
OpMacroTrans<FloatType, MicroType, HEIGHT, WIDTH>::OpMacroTrans(
    const std::shared_ptr<ParamTrans<FloatType>> &param)
    : OpMacro<FloatType, ParamTrans, 1, 1>(param) {
  this->OpMacro::operations[0] = new MicroType(param);
}


template <typename FloatType,
          typename MicroType,
          uint32_t HEIGHT,
          uint32_t WIDTH = HEIGHT>
void OpMacroTrans<FloatType, MicroType, HEIGHT, WIDTH>::exec() {
  // Set operation type alias for UnrollerType
  using OpType = decltype(this->OpMacro::operations);
  // Set unroller type alias for exec_all
  using UnrollerType = RepeatUnrollor<OpType, HEIGHT * WIDTH>;
  // Call micro kernels repeatedly
  this->OpMacro::exec_all<UnrollerType, op_repeat_unroller>();
}

#endif // HPTC_OPERATION_BASE_TCC_