#include <hptc/benchmark/benchmark_trans_avx.h>
#include <hptc/perf-test/test_perf_util.h>

using namespace hptc;


int main() {
  auto ref_func = [] (const float *input_data, float *output_data) {
      sTranspose_3210_608x12x75x96_par<608, 12, 75, 96>(
          input_data, output_data, ALPHA, BETA, nullptr, nullptr);
  };
  auto &ref_config = ref_trans_configs[25];

  compare_perf<float, decltype(ref_func), 4>(ref_func, ref_config);

  return 0;
}
