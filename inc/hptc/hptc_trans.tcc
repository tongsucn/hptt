#pragma once
#ifndef HPTC_HPTC_TRANS_TCC_
#define HPTC_HPTC_TRANS_TCC_

template <typename FloatType,
          TensorUInt ORDER>
CGraphTrans<ParamTrans<TensorWrapper<FloatType, ORDER>>> *
create_cgraph_trans(const FloatType *in_data, FloatType *out_data,
    const std::vector<TensorUInt> &in_size,
    const std::array<TensorUInt, ORDER> &perm,
    const DeducedFloatType<FloatType> alpha,
    const DeducedFloatType<FloatType> beta,
    const TensorUInt num_threads, const TensorInt max_num_cand,
    OuterSize<ORDER> in_outer_size, OuterSize<ORDER> out_outer_size) {
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
  // Create input size objects
  std::array<TensorIdx, ORDER> in_offset, out_offset;
  std::array<TensorIdx, ORDER> in_size_ext, in_outer_size_ext;
  std::copy(in_size.begin(), in_size.end(), in_size_ext.begin());
  if (0 == in_outer_size_vec.size())
    in_outer_size_ext = in_size_ext;
  else {
    std::copy(in_outer_size_vec.begin(), in_outer_size_vec.end(),
        in_outer_size_ext.begin());
    std::copy(in_outer_size.second.begin(), in_outer_size.second.end(),
        in_offset.begin());
  }
  TensorSize<ORDER> in_size_obj(in_size_ext),
      in_outer_size_obj(in_outer_size_ext);

  // Create output size objects
  std::array<TensorIdx, ORDER> out_size_ext, out_outer_size_ext;
  for (auto order_idx = 0; order_idx < ORDER; ++order_idx)
    out_size_ext[order_idx] = in_size_ext[perm[order_idx]];
  if (0 == out_outer_size_vec.size())
    out_outer_size_ext = out_size_ext;
  else {
    std::copy(out_outer_size_vec.begin(), out_outer_size_vec.end(),
        out_outer_size_ext.begin());
    std::copy(out_outer_size.second.begin(), out_outer_size.second.end(),
        out_offset.begin());
  }
  TensorSize<ORDER> out_size_obj(out_size_ext),
      out_outer_size_obj(out_outer_size_ext);

  // Create tensors
  using TensorType = TensorWrapper<FloatType, ORDER>;
  const TensorType in_tensor(in_size_obj, in_outer_size_obj, in_offset,
      in_data);
  TensorType out_tensor(out_size_obj, out_outer_size_obj, out_offset, out_data);

  // Create parameter
  using ParamType = ParamTrans<TensorType>;
  auto param = std::make_shared<ParamType>(in_tensor, out_tensor, perm,
      alpha, beta);

  // Create plan, all heuristics will be generated here
  auto tune_num = max_num_cand < 0 ? -1
      : static_cast<TensorIdx>(std::sqrt(max_num_cand));
  // For now, with this function the library will not generate more than 640
  // (loop order) x 640 (parallelization strategy) candidates
  const TensorIdx heur_num = 640;
  PlanTrans<ParamType> plan(param, tune_num, heur_num, tune_num, heur_num,
      num_threads);

  auto graph =  plan.get_graph();
  return graph;
}

#endif // HPTC_HPTC_TRANS_TCC_
