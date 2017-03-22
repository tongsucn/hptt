#pragma once
#ifndef HPTC_HPTC_TRANS_TCC_
#define HPTC_HPTC_TRANS_TCC_

template <typename FloatType,
          TensorOrder ORDER,
          CoefUsageTrans USAGE>
CGraphTrans<ParamTrans<TensorWrapper<FloatType, ORDER>, USAGE>> *
create_cgraph_trans(const FloatType *in_data, FloatType *out_data,
    const std::vector<TensorOrder> &in_size,
    const std::array<TensorOrder, ORDER> &perm,
    const DeducedFloatType<FloatType> alpha,
    const DeducedFloatType<FloatType> beta, const GenNumType thread_num,
    OuterSize<ORDER> in_outer_size, OuterSize<ORDER> out_outer_size,
    TensorIdx tune_num, TensorIdx heur_num) {
  // Guardian
  // Check template arguments
  if (ORDER <= 1)
    return nullptr;

  // Check raw data
  if (nullptr == in_data or nullptr == out_data)
    return nullptr;

  // Check permutation array
  std::array<bool, ORDER> perm_verify_map;
  perm_verify_map.fill(false);
  for (auto order_idx : perm) {
    if (order_idx >= ORDER or perm_verify_map[order_idx])
      return nullptr;
    else
      perm_verify_map[order_idx] = true;
  }

  // Check tensor sizes
  auto &in_outer_size_vec = in_outer_size.first;
  auto &out_outer_size_vec = out_outer_size.first;
  if (ORDER != in_size.size() or
      (not in_outer_size_vec.empty() and ORDER != in_outer_size_vec.size()) or
      (not out_outer_size_vec.empty() and ORDER != out_outer_size_vec.size()))
    return nullptr;

  // Check outer size values
  for (auto order_idx = 0; order_idx < ORDER; ++order_idx)
    if (0 == in_size[order_idx] or (not in_outer_size_vec.empty() and
        in_outer_size_vec[order_idx] < in_size[order_idx]) or
            (not out_outer_size_vec.empty() and
            out_outer_size_vec[perm[order_idx]] < in_size[order_idx]))
      return nullptr;

  // Check size offset
  if (not in_outer_size_vec.empty())
    for (auto order_idx = 0; order_idx < ORDER; ++order_idx)
      if (in_outer_size.second[order_idx] + in_size[order_idx]
          > in_outer_size_vec[order_idx])
        return nullptr;

  if (not out_outer_size_vec.empty())
    for (auto order_idx = 0; order_idx < ORDER; ++order_idx)
      if (out_outer_size.second[order_idx] + in_size[perm[order_idx]]
          > out_outer_size_vec[order_idx])
        return nullptr;


  // Create transpose plan
  // Create output size and outer size objects
  TensorSize<ORDER> out_size_obj(in_size);
  for (auto order_idx = 0; order_idx < ORDER; ++order_idx)
    out_size_obj[order_idx] = in_size[perm[order_idx]];
  TensorSize<ORDER> out_outer_size_obj(
      0 == out_outer_size_vec.size() ? out_size_obj : out_outer_size_vec);

  if (0 == in_outer_size_vec.size())
    in_outer_size_vec = in_size;
  TensorSize<ORDER> in_size_obj(in_size), in_outer_size_obj(in_outer_size_vec);

  // Create tensors
  using TensorType = TensorWrapper<FloatType, ORDER>;
  const TensorType in_tensor(in_size_obj, in_outer_size_obj,
      in_outer_size.second, in_data);
  TensorType out_tensor(out_size_obj, out_outer_size_obj, out_outer_size.second,
      out_data);

  // Create parameter
  using ParamType = ParamTrans<TensorType, USAGE>;
  auto param = std::make_shared<ParamType>(in_tensor, out_tensor, perm,
      alpha, beta);

  // Create plan, all heuristics will be generated here
  tune_num = tune_num < 0 ? -1 : static_cast<TensorIdx>(std::sqrt(tune_num));
  heur_num = heur_num < 0 ? -1 : static_cast<TensorIdx>(std::sqrt(heur_num));
  PlanTrans<ParamType> plan(param, tune_num, -1, tune_num, -1, thread_num);

  return plan.get_graph();
}

#endif // HPTC_HPTC_TRANS_TCC_
