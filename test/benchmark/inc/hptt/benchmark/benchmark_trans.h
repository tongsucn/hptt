#pragma once
#ifndef HPTT_BENCHMARK_BENCHMARK_TRANS_H_
#define HPTT_BENCHMARK_BENCHMARK_TRANS_H_

#include <vector>

#include <hptt/types.h>


namespace hptt {

struct RefTransConfig {
  RefTransConfig(TensorUInt order, TensorUInt thread_num,
      const std::vector<TensorUInt> &perm,
      const std::vector<TensorIdx> &size);

  TensorUInt order;
  TensorUInt thread_num;
  std::vector<TensorUInt> perm;
  std::vector<TensorIdx> size;
};


std::vector<RefTransConfig> ref_trans_configs {
  RefTransConfig(2, 1, { 1, 0 }, { 1216, 43408 }),
  RefTransConfig(2, 1, { 1, 0 }, { 43408, 1216 }),
  RefTransConfig(2, 1, { 1, 0 }, { 7248, 7248 }),
  RefTransConfig(3, 1, { 0, 2, 1 }, { 2144, 64, 384 }),
  RefTransConfig(3, 1, { 1, 0, 2 }, { 2320, 384, 59 }),
  RefTransConfig(3, 1, { 0, 2, 1 }, { 368, 384, 384 }),
  RefTransConfig(3, 1, { 0, 2, 1 }, { 368, 64, 2307 }),
  RefTransConfig(3, 1, { 1, 0, 2 }, { 384, 384, 355 }),
  RefTransConfig(3, 1, { 1, 0, 2 }, { 384, 2320, 59 }),
  RefTransConfig(3, 1, { 2, 1, 0 }, { 384, 355, 384 }),
  RefTransConfig(3, 1, { 2, 1, 0 }, { 2320, 59, 384 }),
  RefTransConfig(3, 1, { 2, 1, 0 }, { 384, 59, 2320 }),
  RefTransConfig(4, 1, { 0, 3, 2, 1 }, { 80, 96, 75, 96 }),
  RefTransConfig(4, 1, { 0, 3, 2, 1 }, { 464, 16, 75, 96 }),
  RefTransConfig(4, 1, { 0, 3, 2, 1 }, { 80, 16, 75, 582 }),
  RefTransConfig(4, 1, { 2, 1, 3, 0 }, { 96, 75, 96, 75 }),
  RefTransConfig(4, 1, { 2, 1, 3, 0 }, { 608, 12, 96, 75 }),
  RefTransConfig(4, 1, { 2, 1, 3, 0 }, { 96, 12, 608, 75 }),
  RefTransConfig(4, 1, { 2, 0, 3, 1 }, { 96, 75, 96, 75 }),
  RefTransConfig(4, 1, { 2, 0, 3, 1 }, { 608, 12, 96, 75 }),
  RefTransConfig(4, 1, { 2, 0, 3, 1 }, { 96, 12, 608, 75 }),
  RefTransConfig(4, 1, { 1, 0, 3, 2 }, { 96, 96, 75, 75 }),
  RefTransConfig(4, 1, { 1, 0, 3, 2 }, { 608, 96, 12, 75 }),
  RefTransConfig(4, 1, { 1, 0, 3, 2 }, { 96, 608, 12, 75 }),
  RefTransConfig(4, 1, { 3, 2, 1, 0 }, { 96, 75, 75, 96 }),
  RefTransConfig(4, 1, { 3, 2, 1, 0 }, { 608, 12, 75, 96 }),
  RefTransConfig(4, 1, { 3, 2, 1, 0 }, { 96, 12, 75, 608 }),
  RefTransConfig(5, 1, { 0, 4, 2, 1, 3 }, { 32, 48, 28, 28, 48 }),
  RefTransConfig(5, 1, { 0, 4, 2, 1, 3 }, { 176, 8, 28, 28, 48 }),
  RefTransConfig(5, 1, { 0, 4, 2, 1, 3 }, { 32, 8, 28, 28, 298 }),
  RefTransConfig(5, 1, { 3, 2, 1, 4, 0 }, { 48, 28, 28, 48, 28 }),
  RefTransConfig(5, 1, { 3, 2, 1, 4, 0 }, { 352, 4, 28, 48, 28 }),
  RefTransConfig(5, 1, { 3, 2, 1, 4, 0 }, { 48, 4, 28, 352, 28 }),
  RefTransConfig(5, 1, { 2, 0, 4, 1, 3 }, { 48, 28, 48, 28, 28 }),
  RefTransConfig(5, 1, { 2, 0, 4, 1, 3 }, { 352, 4, 48, 28, 28 }),
  RefTransConfig(5, 1, { 2, 0, 4, 1, 3 }, { 48, 4, 352, 28, 28 }),
  RefTransConfig(5, 1, { 1, 3, 0, 4, 2 }, { 48, 48, 28, 28, 28 }),
  RefTransConfig(5, 1, { 1, 3, 0, 4, 2 }, { 352, 48, 4, 28, 28 }),
  RefTransConfig(5, 1, { 1, 3, 0, 4, 2 }, { 48, 352, 4, 28, 28 }),
  RefTransConfig(5, 1, { 4, 3, 2, 1, 0 }, { 48, 28, 28, 28, 48 }),
  RefTransConfig(5, 1, { 4, 3, 2, 1, 0 }, { 352, 4, 28, 28, 48 }),
  RefTransConfig(5, 1, { 4, 3, 2, 1, 0 }, { 48, 4, 28, 28, 352 }),
  RefTransConfig(6, 1, { 0, 3, 2, 5, 4, 1 }, { 16, 32, 15, 32, 15, 15 }),
  RefTransConfig(6, 1, { 0, 3, 2, 5, 4, 1 }, { 48, 10, 15, 32, 15, 15 }),
  RefTransConfig(6, 1, { 0, 3, 2, 5, 4, 1 }, { 16, 10, 15, 103, 15, 15 }),
  RefTransConfig(6, 1, { 3, 2, 0, 5, 1, 4 }, { 32, 15, 15, 32, 15, 15 }),
  RefTransConfig(6, 1, { 3, 2, 0, 5, 1, 4 }, { 112, 5, 15, 32, 15, 15 }),
  RefTransConfig(6, 1, { 3, 2, 0, 5, 1, 4 }, { 32, 5, 15, 112, 15, 15 }),
  RefTransConfig(6, 1, { 2, 0, 4, 1, 5, 3 }, { 32, 15, 32, 15, 15, 15 }),
  RefTransConfig(6, 1, { 2, 0, 4, 1, 5, 3 }, { 112, 5, 32, 15, 15, 15 }),
  RefTransConfig(6, 1, { 2, 0, 4, 1, 5, 3 }, { 32, 5, 112, 15, 15, 15 }),
  RefTransConfig(6, 1, { 3, 2, 5, 1, 0, 4 }, { 32, 15, 15, 32, 15, 15 }),
  RefTransConfig(6, 1, { 3, 2, 5, 1, 0, 4 }, { 112, 5, 15, 32, 15, 15 }),
  RefTransConfig(6, 1, { 3, 2, 5, 1, 0, 4 }, { 32, 5, 15, 112, 15, 15 }),
  RefTransConfig(6, 1, { 5, 4, 3, 2, 1, 0 }, { 32, 15, 15, 15, 15, 32 }),
  RefTransConfig(6, 1, { 5, 4, 3, 2, 1, 0 }, { 112, 5, 15, 15, 15, 32 }),
  RefTransConfig(6, 1, { 5, 4, 3, 2, 1, 0 }, { 32, 5, 15, 15, 15, 112 })
};


/*
 * Import implementation.
 */
#include "benchmark_trans.tcc"

}


#endif // HPTT_BENCHMARK_BENCHMARK_TRANS_H_
