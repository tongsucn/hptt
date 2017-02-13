#include <iostream>

#include <hptc/param/parameter_trans.h>
#include <hptc/benchmark/benchmark_trans_avx.h>
#include <hptc/perf-test/test_perf_util.h>

using namespace std;
using namespace hptc;


int main() {
  auto ref_func = [] (const float *input_data, float *output_data) {
      sTranspose_320514_32x15x15x32x15x1_par<32, 15, 15, 32, 15, 1>(
          input_data, output_data, ALPHA, BETA, nullptr, nullptr);
  };
  auto &ref_config = ref_trans_configs[45];

  compare_perf<float, decltype(ref_func), CoefUsageTrans::USE_BOTH, 6>(ref_func,
      ref_config);

  return 0;
}