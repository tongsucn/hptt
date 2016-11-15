#pragma once
#ifndef HPTC_OPERATION_H_
#define HPTC_OPERATION_H_

#include <cstdint>

#include <memory>
#include <functional>

#include <hptc/types.h>
#include <hptc/util.h>
#include <htpc/parameter.h>

namespace hptc {

template <typename FloatType,
          typename ParamType>
class Operation {
public:
  Operation(const std::shared_ptr<ParamType<FloatType>> &param,
      Operation *prev = nullptr, Operation *next = nullptr);

  Operation(const Operation &operation) = default;
  Operation &operator=(const Operation &operation) = default;

  virtual ~Operation() = default;

  inline void set_prev(Operation *prev);
  inline Operation *get_prev();
  inline void set_next(Operation *next);
  inline Operation *get_next();

  virtual void exec() = 0;

protected:
  std::shared_ptr<ParamType<FloatType>> param;

private:
  Operation *prev;
  Operation *next;
};


template <typename FloatType,
          typename ParamType,
          uint32_t OPER_NUM>
class OpLoop : public Operation<FloatType, ParamType> {
public:
  OpLoop(const std::shared_ptr<ParamType<FloatType>> &param);

  OpLoop(const OpLoop &operation) = default;
  OpLoop &operator=(const OpLoop &operation) = default;

  virtual ~OpLoop() = default;

  virtual void exec() override = 0;

  template <typename OperType>
  inline void init_operation(uint32_t operation_idx,
      const std::shared_ptr<OperType> &oper);

protected:
  inline void exec_all();

private:
  std::shared_ptr<Operation> operations[OPER_NUM];
};


template <typename IdxType>
using ForCondType = std::function<bool(IdxType, IdxType)>;

template <typename IdxType>
struct ForCond {
  constexpr static ForCondType<IdxType> Larger
      = [] (IdxType curr, IdxType guard) -> bool { return curr > guard; };
  constexpr static ForCondType<IdxType> LargerEq
      = [] (IdxType curr, IdxType guard) -> bool { return curr >= guard; };
  constexpr static ForCondType<IdxType> Smaller
      = [] (IdxType curr, IdxType guard) -> bool { return curr < guard; };
  constexpr static ForCondType<IdxType> SmallerEq
      = [] (IdxType curr, IdxType guard) -> bool { return curr <= guard; };
  constexpr static ForCondType<IdxType> Equal
      = [] (IdxType curr, IdxType guard) -> bool { return curr == guard; };
  constexpr static ForCondType<IdxType> NonEqual
      = [] (IdxType curr, IdxType guard) -> bool { return curr != guard; };
};


template <typename FloatType,
          typename ParamType,
          uint32_t OPER_NUM>
class OpLoopFor : public OpLoop<FloatType, ParamType, OPER_NUM> {
public:
  OpLoopFor(const std::shared_ptr<ParamType<FloatType>> &param,
      TensorIdx &target_idx, TensorIdx begin, TensorIdx end, TensorIdx step = 1,
      ForCondType<TensorIdx> cond = ForCond::Smaller);

  OpLoopFor(const OpLoopFor &operation) = default;
  OpLoopFor &operator=(const OpLoopFor &operation) = default;

  virtual ~OpLoopFor() = default;

  virtual inline void exec() final;

private:
  TensorIdx &curr_idx;
  TensorIdx begin, end, step;
  ForCondType<TensorIdx> cond;
};


template <typename FloatType,
          typename ParamType>
class OpMicro : public Operation<FloatType, ParamType> {
public:
  OpMicro(const std::shared_ptr<ParamType<FloatType>> &param);

  OpMicro(const OpMicro &operation) = default;
  OpMicro &operator=(const OpMicro &operation) = default;

  virtual ~OpMicro() = default;

  virtual void exec() override = 0;
};


template <typename FloatType,
          typename ParamType,
          uint32_t HEIGHT,
          uint32_t WIDTH>
class OpMicroCopier : public OpMicro<FloatType, ParamType> {
public:
  OpMicroCopier(const std::shared_ptr<ParamType<FloatType>> &param);

  OpMicroCopier(const OpMicroCopier &operation) = default;
  OpMicroCopier &operator=(const OpMicroCopier &operation) = default;

  virtual ~OpMicroCopier() = default;

  virtual void exec() final;
};


template <typename FloatType,
          typename ParamType,
          uint32_t HEIGHT,
          uint32_t WIDTH>
class OpMacro : public Operation<FloatType, ParamType> {
public:
  OpMacro(const std::shared_ptr<ParamType<FloatType>> &param);

  OpMacro(const OpMacro &operation) = default;
  OpMacro &operator=(const OpMacro &operation) = default;

  virtual ~OpMacro();

  virtual void exec() override = 0;

protected:
  inline void exec_all();

private:
  Operation<FloatType, ParamType> *operations[HEIGHT * WIDTH];
};


template <typename FloatType,
          typename ParamType,
          uint32_t HEIGHT,
          uint32_t WIDTH>
class OpMacroCopier : public OpMacro<FloatType, ParamType, HEIGHT, WIDTH> {
public:
  OpMacroCopier(const std::shared_ptr<ParamType<FloatType>> &param);

  OpMacroCopier(const OpMacroCopier &operation) = default;
  OpMacroCopier &operator=(const OpMacroCopier &operation) = default;

  virtual ~OpMacroCopier() = default;

  virtual void exec() final;
};


/*
 * Import implementation
 */
#include "operation.tcc"


/*
 * Template instantiation typedef for micro/macro kernels
 */
template <typename FloatType,
          typename ParamType = ParamTrans<FloatType>>
using MicroCopier1x1 = MicroCopier<FloatType, ParamType, 1, 1>;


template <typename FloatType,
          typename ParamType = ParamTrans<FloatType>>
using MicroCopier2x2 = MicroCopier<FloatType, ParamType, 2, 2>;


template <typename FloatType,
          typename ParamType = ParamTrans<FloatType>>
using MicroCopier4x4 = MicroCopier<FloatType, ParamType, 4, 4>;


template <typename FloatType,
          typename ParamType = ParamTrans<FloatType>>
using MicroCopier8x8 = MicroCopier<FloatType, ParamType, 8, 8>;


template <typename FloatType,
          typename ParamType = ParamTrans<FloatType>>
using MacroCopier8x16 = MacroCopier<FloatType, ParamType, 8, 16>;

}

#endif // HPTC_OPERATION_H_
