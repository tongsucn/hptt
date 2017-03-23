#pragma once
#ifndef HPTC_BENCHMARK_BENCHMARK_TRANS_H_
#define HPTC_BENCHMARK_BENCHMARK_TRANS_H_

#include <vector>

#include <hptc/types.h>


namespace hptc {

struct RefTransConfig {
  RefTransConfig(TensorUInt order, TensorUInt thread_num,
      const std::vector<TensorUInt> &perm,
      const std::vector<TensorIdx> &size);

  TensorUInt order;
  TensorUInt thread_num;
  std::vector<TensorUInt> perm;
  std::vector<TensorIdx> size;
};

/*
 * Import implementation.
 */
#include "benchmark_trans.tcc"

}


#endif // HPTC_BENCHMARK_BENCHMARK_TRANS_H_
