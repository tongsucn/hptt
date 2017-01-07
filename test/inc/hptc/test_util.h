#pragma once
#ifndef TEST_TEST_UTIL_H_
#define TEST_TEST_UTIL_H_

#include <cmath>

#include <algorithm>

#include <hptc/types.h>

namespace hptc {

template <typename FloatType>
TensorIdx verify(const FloatType *ref_data, const FloatType *act_data,
    TensorIdx data_len);


/*
 * Import implementation.
 */
#include "test_util.tcc"

}

#endif // TEST_TEST_UTIL_H_
