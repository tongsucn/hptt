#pragma once
#ifndef HPTC_PERF_TEST_TEST_PERF_UTIL_H_
#define HPTC_PERF_TEST_TEST_PERF_UTIL_H_

#include <cfloat>

#include <vector>
#include <sstream>
#include <iomanip>
#include <iostream>

#include <hptc/hptc.h>
#include <hptc/test_util.h>
#include <hptc/benchmark/benchmark_trans.h>

#define ALPHA 2.3f
#define BETA 4.2f
#define MEASURE_REPEAT 5


namespace hptc {

template <typename FloatType,
          typename RefFuncType,
          TensorUInt ORDER>
void compare_perf(RefFuncType &ref_func, const RefTransConfig &test_case);


/*
 * Import implementation
 */
#include "test_perf_util.tcc"

}

#endif // HPTC_PERF_TEST_TEST_PERF_UTIL_H_
