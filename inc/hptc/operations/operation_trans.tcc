#pragma once
#ifndef HPTC_OPERATIONS_OPERATION_TRANS_TCC_
#define HPTC_OPERATIONS_OPERATION_TRANS_TCC_

/*
 * Implementation for class OpMicroTrans
 */
template <typename FloatType,
          uint32_t HEIGHT,
          uint32_t WIDTH>
OpMicroTrans<FloatType, HEIGHT, WIDTH>::OpMicroTrans(
    const std::shared_ptr<ParamTrans<FloatType>> &param)
    : OpMicro<FloatType, ParamTrans, HEIGHT, WIDTH>(param) {
  DeducedFloatType<FloatType> alpha = param->alpha, beta = param->beta;
  CoefUsage usage = CoefUsage::USE_NONE;
  if (static_cast<DeducedFloatType<FloatType>>(1) != param->alpha)
    usage |= CoefUsage::USE_ALPHA;
  if (static_cast<DeducedFloatType<FloatType>>(0) != param->beta)
    usage |= CoefUsage::USE_BETA;

  switch (usage) {
  case CoefUsage::NONE:
    this->kernel = new KernelTransAvxDefault<FloatType, CoefUsage::NONE>;
    break;
  case CoefUsage::USE_ALPHA:
    this->kernel = new KernelTransAvxDefault<FloatType, CoefUsage::USE_ALPHA>;
    break;
  case CoefUsage::USE_BETA:
    this->kernel = new KernelTransAvxDefault<FloatType, CoefUsage::USE_BETA>;
    break;
  case CoefUsage::USE_BOTH:
    this->kernel = new KernelTransAvxDefault<FloatType, CoefUsage::USE_BOTH>;
    break;
  default:
    this->kernel = nullptr;
    break;
  }
}


template <typename FloatType,
          uint32_t HEIGHT,
          uint32_t WIDTH>
OpMicroTrans<FloatType, HEIGHT, WIDTH>::~OpMicroTrans() {
  delete kernel;
}


template <typename FloatType,
          uint32_t HEIGHT,
          uint32_t WIDTH>
void OpMicroTrans<FloatType, HEIGHT, WIDTH>::exec() {
}


/*
 * Implementation for class OpMacroTrans
 */
template <typename FloatType,
          typename MicroType,
          uint32_t HEIGHT,
          uint32_t WIDTH>
OpMacroTrans<FloatType, MicroType, HEIGHT, WIDTH>::OpMacroTrans(
    const std::shared_ptr<ParamTrans<FloatType>> &param)
    : OpMacro<FloatType, ParamTrans, 1>(param) {
  this->OpMacro::operations[0] = new MicroType(param);
}


template <typename FloatType,
          typename MicroType,
          uint32_t HEIGHT,
          uint32_t WIDTH>
void OpMacroTrans<FloatType, MicroType, HEIGHT, WIDTH>::exec() {
  // Set operation type alias for UnrollerType
  using OpType = decltype(this->OpMacro::operations);
  // Set unroller type alias for exec_all
  using UnrollerType = RepeatUnrollor<OpType, HEIGHT * WIDTH>;
  // Call micro kernels repeatedly
  this->OpMacro::exec_all<UnrollerType, op_repeat_unroller>();
}

#endif // HPTC_OPERATIONS_OPERATION_TRANS_TCC_
