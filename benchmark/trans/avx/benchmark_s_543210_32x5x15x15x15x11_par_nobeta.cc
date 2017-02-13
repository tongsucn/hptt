#include <iostream>

#include <hptc/param/parameter_trans.h>
#include <hptc/benchmark/benchmark_trans_avx.h>
#include <hptc/perf-test/test_perf_util.h>

using namespace std;
using namespace hptc;


int main() {
  auto ref_func = [] (const float *input_data, float *output_data) {
      sTranspose_543210_32x5x15x15x15x11_par_bz<32, 5, 15, 15, 15, 11>(
          input_data, output_data, ALPHA, nullptr, nullptr);
  };
  auto &ref_config = ref_trans_configs[56];

  compare_perf<float, decltype(ref_func), CoefUsageTrans::USE_ALPHA, 6>(
      ref_func, ref_config);

  return 0;
}