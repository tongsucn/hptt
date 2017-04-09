#include <hptt/benchmark/benchmark_trans.h>
#include <hptt/perf-test/test_perf_util.h>
#include <hptt/test_util.h>

using namespace std;
using namespace hptt;


int main() {
  RefTrans<float> ref_trans;
  print_title("Benchmark for single precision float");

  const auto &configs = ref_trans_configs;
  const auto bm_num = static_cast<TensorInt>(configs.size());
  for (auto idx = 0; idx < bm_num; ++idx) {
    switch (configs[idx].order) {
    case 2:
      compare_perf<float, RefTrans<float>, 2>(ref_trans, configs[idx], idx);
      break;
    case 3:
      compare_perf<float, RefTrans<float>, 3>(ref_trans, configs[idx], idx);
      break;
    case 4:
      compare_perf<float, RefTrans<float>, 4>(ref_trans, configs[idx], idx);
      break;
    case 5:
      compare_perf<float, RefTrans<float>, 5>(ref_trans, configs[idx], idx);
      break;
    case 6:
      compare_perf<float, RefTrans<float>, 6>(ref_trans, configs[idx], idx);
      break;
    }
  }


  return 0;
}
