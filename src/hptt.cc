#include <hptt/hptt.h>

#include <vector>
#include <memory>

#include <hptt/types.h>
#include <hptt/arch/arch.h>
#include <hptt/impl/hptt_trans.h>


namespace hptt {

/*
 * Implementation of function create_cgraph_trans
 */
template <typename FloatType>
std::shared_ptr<CGraphTransPackBase<FloatType>> create_trans_plan(
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

  // Locate function
  using FuncType_ = CGraphTransPackBase<FloatType> *(*)(const FloatType *,
      FloatType *, const TensorUInt, const std::vector<TensorUInt> &,
      const std::vector<TensorUInt> &, const DeducedFloatType<FloatType>,
      const DeducedFloatType<FloatType>, const TensorUInt num_threads,
      const double, const std::vector<TensorUInt> &,
      const std::vector<TensorUInt> &);

  auto &loader = LibLoader::get_loader();

  void *raw_func_trans = nullptr;
  if (std::is_same<float, FloatType>::value) {
    raw_func_trans = loader.dlsym(
        "_Z24create_trans_plan_impl_sPKfPfjRKSt6vectorIjSaIjEES6_ffjdS6_S6_");
  }
  else if (std::is_same<double, FloatType>::value) {
    raw_func_trans = loader.dlsym(
        "_Z24create_trans_plan_impl_dPKdPdjRKSt6vectorIjSaIjEES6_ddjdS6_S6_");
  }
  else if (std::is_same<FloatComplex, FloatType>::value) {
    raw_func_trans = loader.dlsym(
        "_Z24create_trans_plan_impl_cPKCfPS_jRKSt6vectorIjSaIjEES7_ffjdS7_S7_");
  }
  else {
    raw_func_trans = loader.dlsym(
        "_Z24create_trans_plan_impl_zPKCdPS_jRKSt6vectorIjSaIjEES7_ddjdS7_S7_");
  }

  if (nullptr == raw_func_trans)
    return nullptr;
  else
    return std::shared_ptr<CGraphTransPackBase<FloatType>>(
        reinterpret_cast<FuncType_>(raw_func_trans)(in_data, out_data, order,
        in_size, perm, alpha, beta, num_threads, tuning_timeout, in_outer_size,
        out_outer_size));
}


/*
 * Explicit template instantiation definition for function create_trans_plan
 */
template std::shared_ptr<CGraphTransPackBase<float>>
create_trans_plan<float>(const float *, float *,
    const std::vector<TensorUInt> &, const std::vector<TensorUInt> &,
    const DeducedFloatType<float>, const DeducedFloatType<float>,
    const TensorUInt, const double, const std::vector<TensorUInt> &,
    const std::vector<TensorUInt> &);
template std::shared_ptr<CGraphTransPackBase<double>>
create_trans_plan<double>(const double *, double *,
    const std::vector<TensorUInt> &, const std::vector<TensorUInt> &,
    const DeducedFloatType<double>, const DeducedFloatType<double>,
    const TensorUInt, const double, const std::vector<TensorUInt> &,
    const std::vector<TensorUInt> &);
template std::shared_ptr<CGraphTransPackBase<FloatComplex>>
create_trans_plan<FloatComplex>(const FloatComplex *, FloatComplex *,
    const std::vector<TensorUInt> &, const std::vector<TensorUInt> &,
    const DeducedFloatType<FloatComplex>, const DeducedFloatType<FloatComplex>,
    const TensorUInt, const double, const std::vector<TensorUInt> &,
    const std::vector<TensorUInt> &);
template std::shared_ptr<CGraphTransPackBase<DoubleComplex>>
create_trans_plan<DoubleComplex>(const DoubleComplex *, DoubleComplex *,
    const std::vector<TensorUInt> &, const std::vector<TensorUInt> &,
    const DeducedFloatType<DoubleComplex>,
    const DeducedFloatType<DoubleComplex>, const TensorUInt,
    const double, const std::vector<TensorUInt> &,
    const std::vector<TensorUInt> &);

}
