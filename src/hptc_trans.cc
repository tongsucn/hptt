#include <hptc/hptc_trans.h>

#include <cstdint>
#include <vector>

#include <hptc/types.h>


namespace hptc {

/*
 * Implementation for function create_cgraph_trans
 */
template <typename FloatType>
CGraphTransPack<FloatType> *create_trans_plan(
    const FloatType *in_data, FloatType *out_data,
    const std::vector<TensorUInt> &in_size, const std::vector<TensorUInt> &perm,
    const DeducedFloatType<FloatType> alpha,
    const DeducedFloatType<FloatType> beta,
    const TensorUInt num_threads, const double tuning_timeout,
    const std::vector<TensorUInt> &in_outer_size,
    const std::vector<TensorUInt> &out_outer_size) {
  // Guardian
  // Check raw data
  if (nullptr == in_data or nullptr == out_data)
    return nullptr;

  // Check order value
  auto order = static_cast<TensorUInt>(perm.size());
  if (order <= 1)
    return nullptr;

  // Check tensor sizes
  if (order != in_size.size() or
      (not in_outer_size.empty() and order != in_outer_size.size()) or
      (not out_outer_size.empty() and order != out_outer_size.size()))
    return nullptr;

  // Check outer size values
  for (TensorUInt order_idx = 0; order_idx < order; ++order_idx)
    if (0 == in_size[order_idx] or (not in_outer_size.empty() and
        in_outer_size[order_idx] < in_size[order_idx]) or
            (not out_outer_size.empty() and
            out_outer_size[perm[order_idx]] < in_size[order_idx]))
      return nullptr;

  // Check permutation array
  std::vector<bool> perm_verify_map(order, false);
  for (TensorUInt order_idx : perm) {
    if (order_idx >= order or perm_verify_map[order_idx])
      return nullptr;
    else
      perm_verify_map[order_idx] = true;
  }

  // For now, heuristic number will be limited to 1000 (loop orders) x 1000 (
  // parallelization strategies) candidates.
  constexpr auto heur_num = 1000;

  // Set auto-tuning amount and convert timeout from second to millisecond.
  // 64 (loop orders) x 64 (parallelization strategies) will be tuned.
  const auto tune_num = 0.0 == tuning_timeout ? 0 : 64;
  const auto tuning_timeout_ms = tuning_timeout * 1000;

  // Create transpose computational graph package
  return new CGraphTransPack<FloatType>(in_data, out_data, order, in_size,
      perm, alpha, beta, num_threads, tune_num, tune_num, heur_num, heur_num,
      tuning_timeout_ms, in_outer_size, out_outer_size);
}


/*
 * Import explicit template instantiation definition
 */
#include <hptc/gen/hptc_trans_gen.tcc>

}
