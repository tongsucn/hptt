#include <hptt/benchmark/benchmark_trans.h>
#include <hptt/perf-test/test_perf_util.h>
#include <hptt/test_util.h>

using namespace hptt;


int main() {
  RefTrans<float> ref_trans;
  auto &ref_config = ref_trans_configs[50];

  compare_perf<float, RefTrans<float>, 6>(ref_trans, ref_config);

  return 0;
}