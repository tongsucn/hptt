#pragma once
#ifndef HPTT_PERF_TEST_TEST_PERF_UTIL_H_
#define HPTT_PERF_TEST_TEST_PERF_UTIL_H_

#include <cfloat>
#include <cstdio>

#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <iostream>

#include <hptt/hptt.h>
#include <hptt/util/util.h>
#include <hptt/util/util_trans.h>
#include <hptt/test_util.h>
#include <hptt/benchmark/benchmark_trans.h>

#define ALPHA 2.3
#define BETA 4.2
#define MEASURE_REPEAT 10


namespace hptt {

template <typename FloatType,
          typename RefFuncType,
          TensorUInt ORDER>
void compare_perf(RefFuncType &ref_func, const RefTransConfig &test_case,
    bool randomize = false);


void print_title(std::string title);


/*
 * Import implementation
 */
#include "test_perf_util.tcc"

}

#endif // HPTT_PERF_TEST_TEST_PERF_UTIL_H_
