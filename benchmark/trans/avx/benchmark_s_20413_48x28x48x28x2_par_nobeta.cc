#include <iostream>

#include <hptc/param/parameter_trans.h>
#include <hptc/benchmark/benchmark_trans_avx.h>
#include <hptc/perf-test/test_perf_util.h>

using namespace std;
using namespace hptc;


int main() {
  auto ref_func = [] (const float *input_data, float *output_data) {
      sTranspose_20413_48x28x48x28x2_par_bz<48, 28, 48, 28, 2>(
          input_data, output_data, ALPHA, nullptr, nullptr);
  };
  auto &ref_config = ref_trans_configs[33];

  compare_perf<float, decltype(ref_func), CoefUsageTrans::USE_ALPHA, 5>(
      ref_func, ref_config);

  return 0;
}