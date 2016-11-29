#pragma once
#ifndef HPTC_OPERATION_TRANS_H_
#define HPTC_OPERATION_TRANS_H_

#include <hptc/operations/operation_base.h>
#include <hptc/kernels/kernel_trans.h>
#include <hptc/param/parameters.h>

namespace hptc {

template <typename FloatType,
          uint32_t HEIGHT = 1,
          uint32_t WIDTH = HEIGHT>
class OpMicroTrans : public OpMicro<FloatType, ParamTrans, HEIGHT, WIDTH> {
public:
  OpMicroTrans(const std::shared_ptr<ParamTrans<FloatType>> &param);

  OpMicroTrans(const OpMicroTrans &operation) = default;
  OpMicroTrans &operator=(const OpMicroTrans &operation) = delete;

  virtual ~OpMicroTrans() = default;

  virtual void exec() final;

private:
  const FloatType *input_data;
  FloatType *output_data;
  TensorIdx input_offset, output_offset;
  DeducedFloatType<FloatType> alpha, beta;
};


template <typename FloatType,
          typename MicroType,
          uint32_t HEIGHT,
          uint32_t WIDTH = HEIGHT>
class OpMacroTrans : public OpMacro<FloatType, ParamTrans, 1> {
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


/*
 * Transpose Macro kernels template specialization
 */
template <typename FloatType,
          typename MicroType>
using OpMacroTrans1x2 = OpMacroTrans<FloatType, MicroType, 1, 2>;


template <typename FloatType,
          typename MicroType>
using OpMacroTrans1x3 = OpMacroTrans<FloatType, MicroType, 1, 3>;


template <typename FloatType,
          typename MicroType>
using OpMacroTrans1x4 = OpMacroTrans<FloatType, MicroType, 1, 4>;


template <typename FloatType,
          typename MicroType>
using OpMacroTrans2x1 = OpMacroTrans<FloatType, MicroType, 2, 1>;


template <typename FloatType,
          typename MicroType>
using OpMacroTrans2x2 = OpMacroTrans<FloatType, MicroType, 2, 2>;


template <typename FloatType,
          typename MicroType>
using OpMacroTrans2x3 = OpMacroTrans<FloatType, MicroType, 2, 3>;


template <typename FloatType,
          typename MicroType>
using OpMacroTrans2x4 = OpMacroTrans<FloatType, MicroType, 2, 4>;


template <typename FloatType,
          typename MicroType>
using OpMacroTrans3x1 = OpMacroTrans<FloatType, MicroType, 3, 1>;


template <typename FloatType,
          typename MicroType>
using OpMacroTrans3x2 = OpMacroTrans<FloatType, MicroType, 3, 2>;


template <typename FloatType,
          typename MicroType>
using OpMacroTrans3x3 = OpMacroTrans<FloatType, MicroType, 3, 3>;


template <typename FloatType,
          typename MicroType>
using OpMacroTrans3x4 = OpMacroTrans<FloatType, MicroType, 3, 4>;


template <typename FloatType,
          typename MicroType>
using OpMacroTrans4x1 = OpMacroTrans<FloatType, MicroType, 4, 1>;


template <typename FloatType,
          typename MicroType>
using OpMacroTrans4x2 = OpMacroTrans<FloatType, MicroType, 4, 2>;


template <typename FloatType,
          typename MicroType>
using OpMacroTrans4x3 = OpMacroTrans<FloatType, MicroType, 4, 3>;


template <typename FloatType,
          typename MicroType>
using OpMacroTrans4x4 = OpMacroTrans<FloatType, MicroType, 4, 4>;

}

#endif // HPTC_OPERATION_TRANS_H_
