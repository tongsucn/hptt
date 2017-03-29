#include <hptc/benchmark/benchmark_trans_avx.h>
#include <hptc/perf-test/test_perf_util.h>

using namespace hptc;


int main() {
  auto ref_func = [] (const float *input_data, float *output_data) {
      sTranspose_021_368x64x2307_par<368, 64, 2307>(
          input_data, output_data, ALPHA, BETA, nullptr, nullptr);
  };
  auto &ref_config = ref_trans_configs[6];

  compare_perf<float, decltype(ref_func), 3>(ref_func, ref_config);

  return 0;
}
