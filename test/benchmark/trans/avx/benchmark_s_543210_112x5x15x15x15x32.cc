#include <hptc/benchmark/benchmark_trans_avx.h>
#include <hptc/perf-test/test_perf_util.h>

using namespace hptc;


int main() {
  auto ref_func = [] (const float *input_data, float *output_data) {
      sTranspose_543210_112x5x15x15x15x32_par<112, 5, 15, 15, 15, 32>(
          input_data, output_data, ALPHA, BETA, nullptr, nullptr);
  };
  auto &ref_config = ref_trans_configs[55];

  compare_perf<float, decltype(ref_func), 6>(ref_func, ref_config);

  return 0;
}