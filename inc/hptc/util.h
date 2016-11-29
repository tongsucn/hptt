#pragma once
#ifndef HPTC_UTIL_H_
#define HPTC_UTIL_H_

#include <cstdint>

namespace hptc {

using GenNumType = uint32_t;

template <GenNumType UnrollDepth>
struct UnrollControllor {
};


template <typename ArrType, GenNumType UnrollDepth>
inline void op_arr_unroller(ArrType oper, UnrollControllor<UnrollDepth>);


template <typename ArrType>
inline void op_arr_unroller(ArrType oper, UnrollControllor<0>);


template <typename ArrType, GenNumType UnrollDepth>
using ArrUnroller = decltype(op_arr_unroller<ArrType, UnrollDepth>);


template <typename OpType, GenNumType UnrollDepth>
inline void op_repeat_unroller(OpType oper, UnrollControllor<UnrollDepth>);


template <typename OpType>
inline void op_repeat_unroller(OpType oper, UnrollControllor<0>);


template <typename OpType, GenNumType UnrollDepth>
using RepeatUnroller = decltype(op_repeat_unroller<OpType, UnrollDepth>);


template <GenNumType GEN_NUM>
struct GenCounter {
};

template <GenNumType GEN_NUM, typename Func, typename... Args>
inline void general_tiler(GenCounter<GEN_NUM>, Func func, Args... args);


template <typename Func, typename... Args>
inline void general_tiler(GenCounter<0>, Func func, Args... args);


/*
 * Implementation for Unroller
 */
template <typename ArrType, GenNumType UnrollDepth>
inline void op_arr_unroller(ArrType oper, UnrollControllor<UnrollDepth>) {
  op_arr_unroller(oper, UnrollControllor<UnrollDepth - 1>());
  oper[UnrollDepth]->exec();
}


template <typename ArrType>
inline void op_arr_unroller(ArrType oper, UnrollControllor<0>) {
  oper[0]->exec();
}


template <typename OpType, GenNumType UnrollDepth>
inline void op_repeat_unroller(OpType oper, UnrollControllor<UnrollDepth>) {
  op_repeat_unroller(oper, UnrollControllor<UnrollDepth - 1>());
  oper[0]->exec();
}


template <typename OpType>
inline void op_repeat_unroller(OpType oper, UnrollControllor<0>) {
  oper[0]->exec();
}


template <GenNumType GEN_NUM, typename Func, typename... Args>
inline void general_tiler(GenCounter<GEN_NUM>, Func func, Args... args) {
  general_tiler(GenCounter<GEN_NUM - 1>(), func, args...);
  func(args...);
}


template <typename Func, typename... Args>
inline void general_tiler(GenCounter<0>, Func func, Args... args) {
  func(args...);
}

}

#endif // HPTC_UTIL_H_
