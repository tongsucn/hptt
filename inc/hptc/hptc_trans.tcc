#pragma once
#ifndef HPTC_HPTC_TRANS_TCC_
#define HPTC_HPTC_TRANS_TCC_

/*
 * Implementation for class CGraphTransPack
 */
template <typename FloatType>
class CGraphTransPackBase;


template <typename FloatType>
class CGraphTransPack final : public CGraphTransPackBase<FloatType> {
public:
  CGraphTransPack(const FloatType *in_data, FloatType *out_data,
      const TensorUInt order, const std::vector<TensorUInt> &in_size,
      const std::vector<TensorUInt> &perm,
      const DeducedFloatType<FloatType> alpha,
      const DeducedFloatType<FloatType> beta,
      const TensorUInt num_threads, const TensorInt tune_loop_num,
      const TensorInt tune_para_num, const TensorInt heur_loop_num,
      const TensorInt heur_para_num, const double tuning_timeout_ms,
      const std::vector<TensorUInt> &in_outer_size,
      const std::vector<TensorUInt> &out_outer_size)
      : CGraphTransPackBase<FloatType>(in_data, out_data, order, in_size, perm,
        alpha, beta, num_threads, tune_loop_num, tune_para_num, heur_loop_num,
        heur_para_num, tuning_timeout_ms, in_outer_size, out_outer_size) {
  }

  // Copy and move are disabled
  CGraphTransPack(const CGraphTransPack &) = delete;
  CGraphTransPack<FloatType> operator=(const CGraphTransPack &) = delete;
  CGraphTransPack(CGraphTransPack &&) = delete;
  CGraphTransPack<FloatType> operator=(CGraphTransPack &&) = delete;

  HPTC_INL void exec() {
    this->exec_base_();
  }

  HPTC_INL void operator()() {
    this->exec_base_();
  }
};


/*
 * Implementation for function create_cgraph_trans
 */
template <typename FloatType>
CGraphTransPack<FloatType> *create_cgraph_trans(
    const FloatType *in_data, FloatType *out_data, const TensorUInt order,
    const std::vector<TensorUInt> &in_size,
    const std::vector<TensorUInt> &perm,
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
  if (order != perm.size())
    return nullptr;
  std::vector<bool> perm_verify_map(order, false);
  for (TensorUInt order_idx : perm) {
    if (order_idx >= order or perm_verify_map[order_idx])
      return nullptr;
    else
      perm_verify_map[order_idx] = true;
  }

  // For now, heuristic number will be limited to 640 (loop orders) x 640 (
  // parallelization strategies) candidates.
  constexpr auto heur_num = 640;

  // Set auto-tuning amount and convert timeout from second to millisecond.
  const auto tune_num = 0.0 == tuning_timeout ? 0 : -1;
  const auto tuning_timeout_ms = tuning_timeout * 1000;

  // Create transpose computational graph package
  return new CGraphTransPack<FloatType>(in_data, out_data, order, in_size,
      perm, alpha, beta, num_threads, tune_num, tune_num, heur_num, heur_num,
      tuning_timeout_ms, in_outer_size, out_outer_size);
}


/*
 * Import implementations of class CGraphTransPack, this file should be
 * generated by cmake script.
 */
#include <hptc/gen/hptc_trans_gen.tcc>


/*
 * Explicit extern template instantiation declaration for class
 * CGraphTransPackBase
 */
extern template class CGraphTransPackBase<float>;
extern template class CGraphTransPackBase<double>;
extern template class CGraphTransPackBase<FloatComplex>;
extern template class CGraphTransPackBase<DoubleComplex>;


/*
 * Explicit extern template instantiation declaration for class CGraphTransPack
 */
extern template class CGraphTransPack<float>;
extern template class CGraphTransPack<double>;
extern template class CGraphTransPack<FloatComplex>;
extern template class CGraphTransPack<DoubleComplex>;


/*
 * Explicit extern template instantiation declaration for function
 * create_cgraph_trans
 */
extern template CGraphTransPack<float> *create_cgraph_trans<float>(
    const float *, float *, const TensorUInt, const std::vector<TensorUInt> &,
    const std::vector<TensorUInt> &, const DeducedFloatType<float>,
    const DeducedFloatType<float>, const TensorUInt, const double,
    const std::vector<TensorUInt> &, const std::vector<TensorUInt> &);
extern template CGraphTransPack<double> *create_cgraph_trans<double>(
    const double *, double *, const TensorUInt, const std::vector<TensorUInt> &,
    const std::vector<TensorUInt> &, const DeducedFloatType<double>,
    const DeducedFloatType<double>, const TensorUInt, const double,
    const std::vector<TensorUInt> &, const std::vector<TensorUInt> &);
extern template CGraphTransPack<FloatComplex> *
create_cgraph_trans<FloatComplex>(const FloatComplex *, FloatComplex *,
    const TensorUInt, const std::vector<TensorUInt> &,
    const std::vector<TensorUInt> &, const DeducedFloatType<FloatComplex>,
    const DeducedFloatType<FloatComplex>, const TensorUInt, const double,
    const std::vector<TensorUInt> &, const std::vector<TensorUInt> &);
extern template CGraphTransPack<DoubleComplex> *
create_cgraph_trans<DoubleComplex>(const DoubleComplex *, DoubleComplex *,
    const TensorUInt, const std::vector<TensorUInt> &,
    const std::vector<TensorUInt> &, const DeducedFloatType<DoubleComplex>,
    const DeducedFloatType<DoubleComplex>, const TensorUInt, const double,
    const std::vector<TensorUInt> &, const std::vector<TensorUInt> &);

#endif // HPTC_HPTC_TRANS_TCC_
