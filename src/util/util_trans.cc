#include <hptc/util/util_trans.h>

#include <vector>
#include <numeric>
#include <functional>


namespace hptc {

template <typename FloatType,
          CoefUsageTrans USAGE>
double calc_tp_trans(const std::vector<TensorIdx> &size, double time_ms) {
  // Compute moved element number
  const auto moved_num = std::accumulate(size.begin(), size.end(), 1,
      std::multiplies<TensorIdx>());

  // Compute moved size (in bytes)
  const auto moved_size = moved_num * sizeof(FloatType)
      * (CoefUsageTrans::USE_BETA == USAGE or CoefUsageTrans::USE_BOTH == USAGE
          ? 3 : 2);

  // Convert from byte to gigabyte
  auto result = moved_size / static_cast<double>(1024)
      / static_cast<double>(1024) / static_cast<double>(1024);

  // Compute through put
  return result * 1000 / time_ms;
}


/*
 * Explicit template instantiation for function calc_tp_trans
 */
template double calc_tp_trans<float, CoefUsageTrans::USE_ALPHA>(
    const std::vector<TensorIdx> &, double);
template double calc_tp_trans<float, CoefUsageTrans::USE_BETA>(
    const std::vector<TensorIdx> &, double);
template double calc_tp_trans<float, CoefUsageTrans::USE_BOTH>(
    const std::vector<TensorIdx> &, double);
template double calc_tp_trans<float, CoefUsageTrans::USE_NONE>(
    const std::vector<TensorIdx> &, double);
template double calc_tp_trans<double, CoefUsageTrans::USE_ALPHA>(
    const std::vector<TensorIdx> &, double);
template double calc_tp_trans<double, CoefUsageTrans::USE_BETA>(
    const std::vector<TensorIdx> &, double);
template double calc_tp_trans<double, CoefUsageTrans::USE_BOTH>(
    const std::vector<TensorIdx> &, double);
template double calc_tp_trans<double, CoefUsageTrans::USE_NONE>(
    const std::vector<TensorIdx> &, double);
template double calc_tp_trans<FloatComplex, CoefUsageTrans::USE_ALPHA>(
    const std::vector<TensorIdx> &, double);
template double calc_tp_trans<FloatComplex, CoefUsageTrans::USE_BETA>(
    const std::vector<TensorIdx> &, double);
template double calc_tp_trans<FloatComplex, CoefUsageTrans::USE_BOTH>(
    const std::vector<TensorIdx> &, double);
template double calc_tp_trans<FloatComplex, CoefUsageTrans::USE_NONE>(
    const std::vector<TensorIdx> &, double);
template double calc_tp_trans<DoubleComplex, CoefUsageTrans::USE_ALPHA>(
    const std::vector<TensorIdx> &, double);
template double calc_tp_trans<DoubleComplex, CoefUsageTrans::USE_BETA>(
    const std::vector<TensorIdx> &, double);
template double calc_tp_trans<DoubleComplex, CoefUsageTrans::USE_BOTH>(
    const std::vector<TensorIdx> &, double);
template double calc_tp_trans<DoubleComplex, CoefUsageTrans::USE_NONE>(
    const std::vector<TensorIdx> &, double);

}
