#include <hptc/benchmark/benchmark_trans_avx.h>
#include <hptc/perf-test/test_perf_util.h>

using namespace hptc;


int main() {
  auto ref_func = [] (const float *input_data, float *output_data) {
      sTranspose_20413_48x28x48x28x28_par<48, 28, 48, 28, 28>(
          input_data, output_data, ALPHA, BETA, nullptr, nullptr);
  };
  auto &ref_config = ref_trans_configs[33];

  compare_perf<float, decltype(ref_func), 5>(ref_func, ref_config);

  return 0;
}