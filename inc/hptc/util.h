#pragma once
#ifndef HPTC_UTIL_H_
#define HPTC_UTIL_H_

#include <hptc/types.h>


namespace hptc {

template <GenNumType GEN_NUM>
struct GenCounter {
};


template <typename ArrType,
          GenNumType UnrollDepth>
INLINE void op_arr_unroller(ArrType oper, GenCounter<UnrollDepth>);


template <typename ArrType>
INLINE void op_arr_unroller(ArrType oper, GenCounter<0>);


template <typename ArrType,
          GenNumType UnrollDepth>
using ArrUnroller = decltype(op_arr_unroller<ArrType, UnrollDepth>);


/*
 * Implementation for Unroller
 */
template <typename ArrType,
          GenNumType UnrollDepth>
INLINE void op_arr_unroller(ArrType oper, GenCounter<UnrollDepth>) {
  op_arr_unroller(oper, GenCounter<UnrollDepth - 1>());
  oper[UnrollDepth]->exec();
}


template <typename ArrType>
INLINE void op_arr_unroller(ArrType oper, GenCounter<0>) {
  oper[0]->exec();
}

}

#endif // HPTC_UTIL_H_
