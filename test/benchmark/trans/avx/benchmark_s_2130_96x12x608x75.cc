#include <hptc/benchmark/benchmark_trans.h>
#include <hptc/perf-test/test_perf_util.h>
#include <hptc/test_util.h>

using namespace hptc;


int main() {
  RefTrans<float> ref_trans;
  auto &ref_config = ref_trans_configs[17];

  compare_perf<float, RefTrans<float>, 4>(ref_trans, ref_config);

  return 0;
}