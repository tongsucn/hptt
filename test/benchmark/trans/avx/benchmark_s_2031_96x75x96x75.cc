#include <hptc/benchmark/benchmark_trans_avx.h>
#include <hptc/perf-test/test_perf_util.h>

using namespace hptc;


int main() {
  auto ref_func = [] (const float *input_data, float *output_data) {
      sTranspose_2031_96x75x96x75_par<96, 75, 96, 75>(
          input_data, output_data, ALPHA, BETA, nullptr, nullptr);
  };
  auto &ref_config = ref_trans_configs[18];

  compare_perf<float, decltype(ref_func), 4>(ref_func, ref_config);

  return 0;
}
