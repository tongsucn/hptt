#pragma once
#ifndef HPTC_UTIL_H_
#define HPTC_UTIL_H_

#include <cstdint>

#include <hptc/operation.h>

namespace hptc {

template <uint32_t UnrollDepth>
struct UnrollControllor {
};


template <typename ArrType, uint32_t UnrollDepth>
inline void op_arr_unroller(ArrType operations, UnrollControllor<UnrollDepth>);


template <typename ArrType>
inline void op_arr_unroller(ArrType operations, UnrollControllor<0>);


/*
 * Implementation for Unroller
 */
template <typename ArrType, uint32_t UnrollDepth>
inline void op_arr_unroller(ArrType operations, UnrollControllor<UnrollDepth>) {
  op_arr_unroller(operations, UnrollControllor<UnrollDepth - 1>());
  operations[UnrollDepth]->exec();
}


template <typename ArrType>
inline void op_arr_unroller(ArrType operations, UnrollControllor<0>) {
  operations[0]->exec();
}

}

#endif // HPTC_UTIL_H_
