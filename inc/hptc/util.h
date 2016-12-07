#pragma once
#ifndef HPTC_UTIL_H_
#define HPTC_UTIL_H_

#include <hptc/types.h>


namespace hptc {

template <GenNumType GEN_NUM>
struct GenCounter {
};


template <GenNumType GEN_NUM,
          typename Intrin,
          typename... Args>
INLINE void intrin_tiler(GenCounter<GEN_NUM>, Intrin intrinsic, Args... args);


template <typename Intrin,
          typename... Args>
INLINE void intrin_tiler(GenCounter<0>, Intrin intrinsic, Args... args);


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
template <GenNumType GEN_NUM,
          typename Intrin,
          typename... Args>
INLINE void intrin_tiler(GenCounter<GEN_NUM>, Intrin intrinsic, Args... args) {
  intrin_tiler(GenCounter<GEN_NUM - 1>(), intrinsic, args...);
  intrinsic(GEN_NUM, args...);
}


template <typename Intrin,
          typename... Args>
INLINE void intrin_tiler(GenCounter<0>, Intrin intrinsic, Args... args) {
  intrinsic(0, args...);
}


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
