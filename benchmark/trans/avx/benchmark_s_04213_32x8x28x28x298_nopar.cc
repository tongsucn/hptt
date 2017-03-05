#include <hptc/config/config_trans.h>
#include <hptc/benchmark/benchmark_trans_avx.h>
#include <hptc/perf-test/test_perf_util.h>

using namespace hptc;


int main() {
  auto ref_func = [] (const float *input_data, float *output_data) {
      sTranspose_04213_32x8x28x28x298<32, 8, 28, 28, 298>(
          input_data, output_data, ALPHA, BETA, nullptr, nullptr);
  };
  auto &ref_config = ref_trans_configs[29];

  compare_perf<float, decltype(ref_func), CoefUsageTrans::USE_BOTH, 5>(ref_func,
      ref_config);

  return 0;
}