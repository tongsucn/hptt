#pragma once
#ifndef HPTC_PERF_TEST_TEST_PERF_UTIL_H_
#define HPTC_PERF_TEST_TEST_PERF_UTIL_H_

#include <array>
#include <memory>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <algorithm>

#include <hptc/types.h>
#include <hptc/util.h>
#include <hptc/tensor.h>
#include <hptc/plan/plan_trans.h>
#include <hptc/param/parameter_trans.h>

#define ALPHA 2.3f
#define BETA 4.2f


namespace hptc {

template <typename FloatType,
          typename RefFuncType,
          CoefUsageTrans USAGE,
          TensorOrder ORDER>
void compare_perf(RefFuncType &ref_func, const RefTransConfig &test_case);


/*
 * Import implementation
 */
#include "test_perf_util.tcc"

}

#endif // HPTC_PERF_TEST_TEST_PERF_UTIL_H_
