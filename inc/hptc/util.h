#pragma once
#ifndef HPTC_UTIL_H_
#define HPTC_UTIL_H_

#include <cstdint>

namespace hptc {

template <uint32_t UnrollDepth>
struct UnrollControllor {
};


template <typename ArrType, uint32_t UnrollDepth>
inline void op_arr_unroller(ArrType oper, UnrollControllor<UnrollDepth>);


template <typename ArrType>
inline void op_arr_unroller(ArrType oper, UnrollControllor<0>);


template <typename ArrType, uint32_t UnrollDepth>
using ArrUnroller = decltype(op_arr_unroller<ArrType, UnrollDepth>);


template <typename OpType, uint32_t UnrollDepth>
inline void op_repeat_unroller(OpType oper, UnrollControllor<UnrollDepth>);


template <typename OpType>
inline void op_repeat_unroller(OpType oper, UnrollControllor<0>);


template <typename OpType, uint32_t UnrollDepth>
using RepeatUnroller = decltype(op_repeat_unroller<OpType, UnrollDepth>);


/*
 * Implementation for Unroller
 */
template <typename ArrType, uint32_t UnrollDepth>
inline void op_arr_unroller(ArrType oper, UnrollControllor<UnrollDepth>) {
  op_arr_unroller(oper, UnrollControllor<UnrollDepth - 1>());
  oper[UnrollDepth]->exec();
}


template <typename ArrType>
inline void op_arr_unroller(ArrType oper, UnrollControllor<0>) {
  oper[0]->exec();
}


template <typename OpType, uint32_t UnrollDepth>
inline void op_repeat_unroller(OpType oper, UnrollControllor<UnrollDepth>) {
  op_repeat_unroller(oper, UnrollControllor<UnrollDepth - 1>());
  oper[0]->exec();
}


template <typename OpType>
inline void op_repeat_unroller(OpType oper, UnrollControllor<0>) {
  oper[0]->exec();
}

}

#endif // HPTC_UTIL_H_
