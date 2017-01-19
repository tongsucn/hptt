#pragma once
#ifndef HPTC_BENCHMARK_BENCHMARK_TRANS_AVX_H_
#define HPTC_BENCHMARK_BENCHMARK_TRANS_AVX_H_

#include <vector>

#include <hptc/types.h>
#include <hptc/compat.h>
#include <hptc/param/parameter_trans.h>

#include <hptc/test_util.h>

#include "trans/avx/benchmark_nopar_beta.h"
#include "trans/avx/benchmark_nopar_nobeta.h"
#include "trans/avx/benchmark_par_beta.h"
#include "trans/avx/benchmark_par_nobeta.h"


namespace hptc {

std::vector<RefTransConfig> ref_trans_configs {
  RefTransConfig(2, 1, { 1, 0 }, { 7248, 724 }),
  RefTransConfig(2, 1, { 1, 0 }, { 1216, 4340 }),
  RefTransConfig(2, 1, { 1, 0 }, { 43408, 121 }),
  RefTransConfig(3, 1, { 0, 2, 1 }, { 2144, 64, 38 }),
  RefTransConfig(3, 1, { 0, 2, 1 }, { 368, 384, 38 }),
  RefTransConfig(3, 1, { 0, 2, 1 }, { 368, 64, 230 }),
  RefTransConfig(3, 1, { 1, 0, 2 }, { 2320, 384, 5 }),
  RefTransConfig(3, 1, { 1, 0, 2 }, { 384, 2320, 5 }),
  RefTransConfig(3, 1, { 1, 0, 2 }, { 384, 384, 35 }),
  RefTransConfig(3, 1, { 2, 1, 0 }, { 2320, 59, 38 }),
  RefTransConfig(3, 1, { 2, 1, 0 }, { 384, 355, 38 }),
  RefTransConfig(3, 1, { 2, 1, 0 }, { 384, 59, 232 }),
  RefTransConfig(4, 1, { 0, 3, 2, 1 }, { 464, 16, 75, 9 }),
  RefTransConfig(4, 1, { 0, 3, 2, 1 }, { 80, 16, 75, 58 }),
  RefTransConfig(4, 1, { 0, 3, 2, 1 }, { 80, 96, 75, 9 }),
  RefTransConfig(4, 1, { 2, 1, 3, 0 }, { 608, 12, 96, 7 }),
  RefTransConfig(4, 1, { 2, 1, 3, 0 }, { 96, 12, 608, 7 }),
  RefTransConfig(4, 1, { 2, 1, 3, 0 }, { 96, 75, 96, 7 }),
  RefTransConfig(4, 1, { 2, 0, 3, 1 }, { 608, 12, 96, 7 }),
  RefTransConfig(4, 1, { 2, 0, 3, 1 }, { 96, 12, 608, 7 }),
  RefTransConfig(4, 1, { 2, 0, 3, 1 }, { 96, 75, 96, 7 }),
  RefTransConfig(4, 1, { 1, 0, 3, 2 }, { 608, 96, 12, 7 }),
  RefTransConfig(4, 1, { 1, 0, 3, 2 }, { 96, 608, 12, 7 }),
  RefTransConfig(4, 1, { 1, 0, 3, 2 }, { 96, 96, 75, 7 }),
  RefTransConfig(4, 1, { 3, 2, 1, 0 }, { 608, 12, 75, 9 }),
  RefTransConfig(4, 1, { 3, 2, 1, 0 }, { 96, 12, 75, 60 }),
  RefTransConfig(4, 1, { 3, 2, 1, 0 }, { 96, 75, 75, 9 }),
  RefTransConfig(5, 1, { 0, 4, 2, 1, 3 }, { 176, 8, 28, 28, 4 }),
  RefTransConfig(5, 1, { 0, 4, 2, 1, 3 }, { 32, 48, 28, 28, 4 }),
  RefTransConfig(5, 1, { 0, 4, 2, 1, 3 }, { 32, 8, 28, 28, 29 }),
  RefTransConfig(5, 1, { 3, 2, 1, 4, 0 }, { 352, 4, 28, 48, 2 }),
  RefTransConfig(5, 1, { 3, 2, 1, 4, 0 }, { 48, 28, 28, 48, 2 }),
  RefTransConfig(5, 1, { 3, 2, 1, 4, 0 }, { 48, 4, 28, 352, 2 }),
  RefTransConfig(5, 1, { 2, 0, 4, 1, 3 }, { 352, 4, 48, 28, 2 }),
  RefTransConfig(5, 1, { 2, 0, 4, 1, 3 }, { 48, 28, 48, 28, 2 }),
  RefTransConfig(5, 1, { 2, 0, 4, 1, 3 }, { 48, 4, 352, 28, 2 }),
  RefTransConfig(5, 1, { 1, 3, 0, 4, 2 }, { 352, 48, 4, 28, 2 }),
  RefTransConfig(5, 1, { 1, 3, 0, 4, 2 }, { 48, 352, 4, 28, 2 }),
  RefTransConfig(5, 1, { 1, 3, 0, 4, 2 }, { 48, 48, 28, 28, 2 }),
  RefTransConfig(5, 1, { 4, 3, 2, 1, 0 }, { 352, 4, 28, 28, 4 }),
  RefTransConfig(5, 1, { 4, 3, 2, 1, 0 }, { 48, 28, 28, 28, 4 }),
  RefTransConfig(5, 1, { 4, 3, 2, 1, 0 }, { 48, 4, 28, 28, 35 }),
  RefTransConfig(6, 1, { 0, 3, 2, 5, 4, 1 }, { 16, 10, 15, 103, 15, 1 }),
  RefTransConfig(6, 1, { 0, 3, 2, 5, 4, 1 }, { 16, 32, 15, 32, 15, 1 }),
  RefTransConfig(6, 1, { 0, 3, 2, 5, 4, 1 }, { 48, 10, 15, 32, 15, 1 }),
  RefTransConfig(6, 1, { 2, 0, 4, 1, 5, 3 }, { 112, 5, 32, 15, 15, 1 }),
  RefTransConfig(6, 1, { 2, 0, 4, 1, 5, 3 }, { 32, 15, 32, 15, 15, 1 }),
  RefTransConfig(6, 1, { 2, 0, 4, 1, 5, 3 }, { 32, 5, 112, 15, 15, 1 }),
  RefTransConfig(6, 1, { 3, 2, 0, 5, 1, 4 }, { 112, 5, 15, 32, 15, 1 }),
  RefTransConfig(6, 1, { 3, 2, 0, 5, 1, 4 }, { 32, 15, 15, 32, 15, 1 }),
  RefTransConfig(6, 1, { 3, 2, 0, 5, 1, 4 }, { 32, 5, 15, 112, 15, 1 }),
  RefTransConfig(6, 1, { 3, 2, 5, 1, 0, 4 }, { 112, 5, 15, 32, 15, 1 }),
  RefTransConfig(6, 1, { 3, 2, 5, 1, 0, 4 }, { 32, 15, 15, 32, 15, 1 }),
  RefTransConfig(6, 1, { 3, 2, 5, 1, 0, 4 }, { 32, 5, 15, 112, 15, 1 }),
  RefTransConfig(6, 1, { 5, 4, 3, 2, 1, 0 }, { 112, 5, 15, 15, 15, 3 }),
  RefTransConfig(6, 1, { 5, 4, 3, 2, 1, 0 }, { 32, 15, 15, 15, 15, 3 }),
  RefTransConfig(6, 1, { 5, 4, 3, 2, 1, 0 }, { 32, 5, 15, 15, 15, 11 })
};

}


#endif // HPTC_BENCHMARK_BENCHMARK_TRANS_AVX_H_
