#pragma once
#ifndef HPTC_UTIL_H_
#define HPTC_UTIL_H_

#include <hptc/types.h>


namespace hptc {

template <GenNumType GEN_NUM>
struct GenCounter {
};


template <GenNumType ROWS,
          GenNumType COLS>
struct DualCounter {
};

}

#endif // HPTC_UTIL_H_
