#include <iostream>

#include <hptc/param/parameter_trans.h>
#include <hptc/benchmark/benchmark_trans_avx.h>
#include <hptc/perf-test/test_perf_util.h>

using namespace std;
using namespace hptc;


int main() {
  auto ref_func = [] (const float *input_data, float *output_data) {
      sTranspose_1032_96x96x75x7_par_bz<96, 96, 75, 7>(
          input_data, output_data, ALPHA, nullptr, nullptr);
  };
  auto &ref_config = ref_trans_configs[21];

  compare_perf<float, decltype(ref_func), CoefUsageTrans::USE_ALPHA, 4>(
      ref_func, ref_config);

  return 0;
}