#pragma once
#ifndef HPTC_OPERATION_TRANS_H_
#define HPTC_OPERATION_TRANS_H_

#include <hptc/operations/operation_base.h>
#include <hptc/kernels/kernel_trans.h>
#include <hptc/param/parameters.h>

namespace hptc {

template <typename FloatType,
          typename ParamType,
          uint32_t HEIGHT,
          uint32_t WIDTH = HEIGHT>
class OpMicroTrans : public OpMicro<FloatType, ParamType, HEIGHT, WIDTH> {
public:
  OpMicroTrans(const std::shared_ptr<ParamType<FloatType>> &param);

  OpMicroTrans(const OpMicroTrans &operation) = default;
  OpMicroTrans &operator=(const OpMicroTrans &operation) = default;

  virtual ~OpMicroTrans() = default;

  virtual void exec() final;

protected:
  virtual void exec_internal() final;
};


template <typename FloatType,
          typename MicroType,
          uint32_t HEIGHT,
          uint32_t WIDTH = HEIGHT>
class OpMacroTrans : public OpMacro<FloatType, ParamTrans, 1, 1> {
public:
  OpMacroTrans(const std::shared_ptr<ParamTrans<FloatType>> &param);

  OpMacroTrans(const OpMacroTrans &operation) = delete;
  OpMacroTrans &operator=(const OpMacroTrans &operation) = delete;

  virtual ~OpMacroTrans() = default;

  virtual void exec() final;
};


/*
 * Import implementation
 */
#include "operation_trans.tcc"

}

#endif // HPTC_OPERATION_TRANS_H_
