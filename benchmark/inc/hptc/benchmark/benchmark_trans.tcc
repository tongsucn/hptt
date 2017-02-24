#pragma once
#ifndef HPTC_BENCHMARK_BENCHMARK_TRANS_TCC_
#define HPTC_BENCHMARK_BENCHMARK_TRANS_TCC_

RefTransConfig::RefTransConfig(TensorOrder order, GenNumType thread_num,
    const std::vector<TensorOrder> &perm, const std::vector<TensorOrder> &size)
    : order(order),
      thread_num(thread_num),
      perm(perm),
      size(size) {
}

#endif // HPTC_BENCHMARK_BENCHMARK_TRANS_TCC_
