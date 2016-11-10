#pragma once
#ifndef HPTC_OPERATION_H_
#define HPTC_OPERATION_H_

#include <cstdint>

#include <memory>
#include <functional>

#include <hptc/types.h>
#include <htpc/parameter.h>

namespace hptc {

class Operation {
public:
  Operation(const std::shared_ptr<Param> &param, Operation *prev = nullptr,
      Operation *next = nullptr);

  Operation(const Operation &operation) = default;
  Operation &operator=(const Operation &operation) = default;

  virtual ~Operation() = default;

  inline void set_prev(Operation *prev);
  inline Operation *get_prev();
  inline void set_next(Operation *next);
  inline Operation *get_next();

  virtual void exec() = 0;

protected:
  std::shared_ptr<Param> param;

private:
  Operation *prev;
  Operation *next;
};


template <uint32_t OPER_NUM>
class OpLoop : public Operation {
public:
  OpLoop(const std::shared_ptr<Param> &param);

  OpLoop(const OpLoop &operation) = default;
  OpLoop &operator=(const OpLoop &operation) = default;

  virtual ~OpLoop() = default;

  template <typename OperType>
  inline void init_operation(uint32_t nth_operation,
      const std::shared_ptr<OperType> &oper);

  virtual void exec() override = 0;

protected:
  inline void exec_all();

private:
  std::shared_ptr<Operation> operations[OPER_NUM];

  template <uint32_t UnrollDepth>
  struct UnrollControllor {
  };

  template <uint32_t UnrollDepth>
  inline void unroller(UnrollControllor<UnrollDepth>);
  inline void unroller(UnrollControllor<0>);
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


template <typename ParamType,
          uint32_t OPER_NUM,
          typename IdxType = TensorIdx>
class OpLoopFor : public OpLoop<OPER_NUM> {
public:
  OpLoopFor(IdxType begin, IdxType end, IdxType step, ForCondType<IdxType> cond,
      const std::shared_ptr<ParamType> &param);

  OpLoopFor(const OpLoopFor &operation) = default;
  OpLoopFor &operator=(const OpLoopFor &operation) = default;

  virtual ~OpLoopFor() = default;

  virtual inline void exec() final;

private:
  IdxType begin, end, step;
  ForCondType<IdxType> cond;
};


class OpMicro : public Operation {
public:
  OpMicro(const std::shared_ptr<Param> &param);

  OpMicro(const OpMicro &operation) = default;
  OpMicro &operator=(const OpMicro &operation) = default;

  virtual ~OpMicro() = default;

  virtual void exec() override = 0;
};


class OpMicroCopier : public OpMicro {
public:
  OpMicroCopier(const std::shared_ptr<Param> &param);

  OpMicroCopier(const OpMicroCopier &operation) = default;
  OpMicroCopier &operator=(const OpMicroCopier &operation) = default;

  virtual ~OpMicroCopier() = default;

  virtual void exec() override = 0;
};


template <typename ParamType>
class OpMicroCopier1x1 : public OpMicroCopier {
public:
  OpMicroCopier1x1(const std::shared_ptr<ParamType> &param);

  OpMicroCopier1x1(const OpMicroCopier1x1 &operation) = default;
  OpMicroCopier1x1 &operator=(const OpMicroCopier1x1 &operation) = default;

  virtual ~OpMicroCopier1x1() = default;

  virtual void exec() final;
};


template <typename ParamType>
class OpMicroCopier2x2 : public OpMicroCopier {
public:
  OpMicroCopier2x2(const std::shared_ptr<ParamType> &param);

  OpMicroCopier2x2(const OpMicroCopier2x2 &operation) = default;
  OpMicroCopier2x2 &operator=(const OpMicroCopier2x2 &operation) = default;

  virtual ~OpMicroCopier2x2() = default;

  virtual void exec() final;
};


template <typename ParamType>
class OpMicroCopier4x4 : public OpMicroCopier {
public:
  OpMicroCopier4x4(const std::shared_ptr<ParamType> &param);

  OpMicroCopier4x4(const OpMicroCopier4x4 &operation) = default;
  OpMicroCopier4x4 &operator=(const OpMicroCopier4x4 &operation) = default;

  virtual ~OpMicroCopier4x4() = default;

  virtual void exec() final;
};


template <typename ParamType>
class OpMicroCopier8x8 : public OpMicroCopier {
public:
  OpMicroCopier8x8(const std::shared_ptr<ParamType> &param);

  OpMicroCopier8x8(const OpMicroCopier8x8 &operation) = default;
  OpMicroCopier8x8 &operator=(const OpMicroCopier8x8 &operation) = default;

  virtual ~OpMicroCopier8x8() = default;

  virtual void exec() final;
};


class OpMacro : public Operation {
public:
  OpMacro(const std::shared_ptr<Param> &param);

  OpMacro(const OpMacro &operation) = default;
  OpMacro &operator=(const OpMacro &operation) = default;

  virtual ~OpMacro() = default;

  virtual void exec() override = 0;
};


class OpMacroCopier : public OpMacro {
public:
  OpMacroCopier(const std::shared_ptr<Param> &param);

  OpMacroCopier(const OpMacroCopier &operation) = default;
  OpMacroCopier &operator=(const OpMacroCopier &operation) = default;

  virtual ~OpMacroCopier() = default;

  virtual void exec() override = 0;
};


class OpMacroCopier8x16 : public OpMacroCopier {
public:
  OpMacroCopier8x16(const std::shared_ptr<Param> &param);

  OpMacroCopier8x16(const OpMacroCopier8x16 &operation) = default;
  OpMacroCopier8x16 &operator=(const OpMacroCopier8x16 &operation) = default;

  virtual ~OpMacroCopier8x16() = default;

  virtual void exec() final;
};


class OpMacroCopier16x8 : public OpMacroCopier {
public:
  OpMacroCopier16x8(const std::shared_ptr<Param> &param);

  OpMacroCopier16x8(const OpMacroCopier16x8 &operation) = default;
  OpMacroCopier16x8 &operator=(const OpMacroCopier16x8 &operation) = default;

  virtual ~OpMacroCopier() = default;

  virtual void exec() final;
};


class OpMacroCopier16x16 : public OpMacroCopier {
public:
  OpMacroCopier16x16(const std::shared_ptr<Param> &param);

  OpMacroCopier16x16(const OpMacroCopier16x16 &operation) = default;
  OpMacroCopier16x16 &operator=(const OpMacroCopier16x16 &operation) = default;

  virtual ~OpMacroCopier16x16() = default;

  virtual void exec() final;
};


class OpMacroCopier16x32 : public OpMacroCopier {
public:
  OpMacroCopier16x32(const std::shared_ptr<Param> &param);

  OpMacroCopier16x32(const OpMacroCopier16x32 &operation) = default;
  OpMacroCopier16x32 &operator=(const OpMacroCopier16x32 &operation) = default;

  virtual ~OpMacroCopier16x32() = default;

  virtual void exec() final;
};


class OpMacroCopier32x16 : public OpMacroCopier {
public:
  OpMacroCopier32x16(const std::shared_ptr<Param> &param);

  OpMacroCopier32x16(const OpMacroCopier32x16 &operation) = default;
  OpMacroCopier32x16 &operator=(const OpMacroCopier32x16 &operation) = default;

  virtual ~OpMacroCopier32x16() = default;

  virtual void exec() final;
};


class OpMacroCopier32x32 : public OpMacroCopier {
public:
  OpMacroCopier32x32(const std::shared_ptr<Param> &param);

  OpMacroCopier32x32(const OpMacroCopier32x32 &operation) = default;
  OpMacroCopier32x32 &operator=(const OpMacroCopier32x32 &operation) = default;

  virtual ~OpMacroCopier32x32() = default;

  virtual void exec() final;
};


// Import implementation
#include "operation.tcc"

}

#endif // HPTC_OPERATION_H_
