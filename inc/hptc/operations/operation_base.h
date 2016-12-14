#pragma once
#ifndef HPTC_OPERATIONS_OPERATION_BASE_H_
#define HPTC_OPERATIONS_OPERATION_BASE_H_

#include <cstdint>

#include <memory>
#include <functional>

#include <hptc/types.h>
#include <hptc/util.h>


namespace hptc {

template <typename FloatType,
          template <typename = FloatType> class ParamType>
class Operation {
public:
  Operation(const std::shared_ptr<ParamType<FloatType>> &param,
      Operation *prev = nullptr, Operation *next = nullptr);

  Operation(const Operation &operation) = default;
  Operation &operator=(const Operation &operation) = default;

  virtual ~Operation() = default;

  INLINE void set_prev(Operation *prev);
  INLINE Operation<FloatType, ParamType> *get_prev();
  INLINE void set_next(Operation *next);
  INLINE Operation<FloatType, ParamType> *get_next();

  virtual void exec() = 0;

protected:
  std::shared_ptr<ParamType<FloatType>> param;
  Operation *prev;
  Operation *next;
};


template <typename FloatType,
          template <typename> typename ParamType,
          GenNumType OPER_NUM>
class OpLoop : public Operation<FloatType, ParamType> {
public:
  OpLoop(const std::shared_ptr<ParamType<FloatType>> &param);

  OpLoop(const OpLoop &operation) = default;
  OpLoop &operator=(const OpLoop &operation) = default;

  virtual ~OpLoop() = default;

  virtual void exec() override = 0;

  template <typename OperType>
  INLINE void init_operation(const std::shared_ptr<OperType> &oper,
      GenNumType operation_idx = 0);

protected:
  INLINE void exec_all();

private:
  std::shared_ptr<Operation<FloatType, ParamType>> operations_[OPER_NUM];
};


using ForCondType = std::function<bool(TensorIdx, TensorIdx)>;

namespace ForCond {
ForCondType Larger
  = [] (TensorIdx curr, TensorIdx guard) -> bool { return curr > guard; };
ForCondType LargerEq
  = [] (TensorIdx curr, TensorIdx guard) -> bool { return curr >= guard; };
ForCondType Smaller
  = [] (TensorIdx curr, TensorIdx guard) -> bool { return curr < guard; };
ForCondType SmallerEq
  = [] (TensorIdx curr, TensorIdx guard) -> bool { return curr <= guard; };
ForCondType Equal
  = [] (TensorIdx curr, TensorIdx guard) -> bool { return curr == guard; };
ForCondType NonEqual
  = [] (TensorIdx curr, TensorIdx guard) -> bool { return curr != guard; };
}

template <typename FloatType,
          template <typename> typename ParamType,
          GenNumType OPER_NUM>
class OpLoopFor : public OpLoop<FloatType, ParamType, OPER_NUM> {
public:
  OpLoopFor(const std::shared_ptr<ParamType<FloatType>> &param,
      TensorIdx &target_idx, TensorIdx begin, TensorIdx end, TensorIdx step,
      ForCondType cond = ForCond::Smaller);

  OpLoopFor(const OpLoopFor &operation) = default;
  OpLoopFor &operator=(const OpLoopFor &operation) = default;

  virtual ~OpLoopFor() = default;

  virtual INLINE void exec() final;

private:
  TensorIdx &curr_idx_;
  TensorIdx begin_, end_, step_;
  ForCondType cond_;
};


template <typename FloatType,
          template <typename> typename ParamType>
class OpMacro : public Operation<FloatType, ParamType> {
public:
  OpMacro(const std::shared_ptr<ParamType<FloatType>> &param);

  OpMacro(const OpMacro &operation) = delete;
  OpMacro &operator=(const OpMacro &operation) = delete;

  virtual ~OpMacro() = default;

  virtual void exec() override = 0;
};


/*
 * Import implementation
 */
#include "operation_base.tcc"

}

#endif // HPTC_OPERATIONS_OPERATION_BASE_H_
