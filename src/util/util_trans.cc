#include <hptc/util/util_trans.h>

#include <vector>
#include <numeric>
#include <functional>

#include <hptc/types.h>


namespace hptc {

template <typename FloatType>
double calc_tp_trans(const std::vector<TensorIdx> &size, double time_ms) {
  // Compute moved element number
  const auto moved_num = std::accumulate(size.begin(), size.end(), 1,
      std::multiplies<TensorIdx>());

  // Compute moved size (in bytes)
  const auto moved_size = moved_num * sizeof(FloatType) * 3;

  // Convert from byte to gigabyte
  auto result = moved_size / static_cast<double>(1024)
      / static_cast<double>(1024) / static_cast<double>(1024);

  // Compute through put
  return result * 1000 / time_ms;
}


/*
 * Explicit template instantiation for function calc_tp_trans
 */
template double calc_tp_trans<float>(const std::vector<TensorIdx> &, double);
template double calc_tp_trans<double>(const std::vector<TensorIdx> &, double);
template double calc_tp_trans<FloatComplex>(
    const std::vector<TensorIdx> &, double);
template double calc_tp_trans<DoubleComplex>(
    const std::vector<TensorIdx> &, double);

}
