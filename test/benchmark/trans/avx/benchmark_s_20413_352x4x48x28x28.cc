#include <hptc/benchmark/benchmark_trans_avx.h>
#include <hptc/perf-test/test_perf_util.h>

using namespace hptc;


int main() {
  auto ref_func = [] (const float *input_data, float *output_data) {
      sTranspose_20413_352x4x48x28x28_par<352, 4, 48, 28, 28>(
          input_data, output_data, ALPHA, BETA, nullptr, nullptr);
  };
  auto &ref_config = ref_trans_configs[34];

  compare_perf<float, decltype(ref_func), 5>(ref_func, ref_config);

  return 0;
}
