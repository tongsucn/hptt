#pragma once
#ifndef HPTC_BENCHMARK_BENCHMARK_TRANS_H_
#define HPTC_BENCHMARK_BENCHMARK_TRANS_H_

#include <vector>

#include <hptc/types.h>


namespace hptc {

struct RefTransConfig {
  RefTransConfig(TensorOrder order, GenNumType thread_num,
      const std::vector<TensorOrder> &perm,
      const std::vector<TensorOrder> &size);

  TensorOrder order;
  GenNumType thread_num;
  std::vector<TensorOrder> perm;
  std::vector<TensorOrder> size;
};

/*
 * Import implementation.
 */
#include "benchmark_trans.tcc"

}


#endif // HPTC_BENCHMARK_BENCHMARK_TRANS_H_
