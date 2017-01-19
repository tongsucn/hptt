#pragma once
#ifndef HPTC_PERF_TEST_TEST_PERF_UTIL_TCC_
#define HPTC_PERF_TEST_TEST_PERF_UTIL_TCC_

template <typename FloatType,
          typename RefFuncType,
          CoefUsage USAGE,
          TensorOrder ORDER>
void compare_perf(RefFuncType &ref_func, const RefTransConfig &test_case) {
  using Deduced = DeducedFloatType<FloatType>;

  DataWrapper<FloatType> data_wrapper(test_case.size);
  data_wrapper.reset_ref();
  data_wrapper.reset_act();
  TimerWrapper timer(20);

  // Measure TTC version
  double ttc_time = timer(ref_func, data_wrapper.org_in_data,
      data_wrapper.ref_data);

  // Measure HPTC version
  // Create tensor wrapper and parameters
  // 1. Create size objects
  TensorSize<ORDER> input_size(test_case.size);
  auto output_size_vec = test_case.size;
  for (TensorOrder idx = 0; idx < ORDER; ++idx)
    output_size_vec[idx] = input_size[test_case.perm[idx]];
  TensorSize<ORDER> output_size(output_size_vec);

  // 2. Create tensor wrappers
  TensorWrapper<FloatType, ORDER> input_tensor(input_size,
      data_wrapper.org_in_data);
  TensorWrapper<FloatType, ORDER> output_tensor(output_size,
      data_wrapper.act_data);

  // 3. Create parameter
  std::array<TensorOrder, ORDER> perm;
  copy(test_case.perm.begin(), test_case.perm.end(), perm.begin());
  auto param = std::make_shared<ParamTrans<FloatType, ORDER, USAGE>>(
      input_tensor, output_tensor, perm, static_cast<Deduced>(ALPHA),
      static_cast<Deduced>(BETA));

  // 4. Create kernels
  TensorOrder input_leading
      = param->input_tensor.get_size()[ORDER - param->merged_order];
  TensorOrder output_leading
      = param->output_tensor.get_size()[ORDER - param->merged_order];

  MacroTransVec<FloatType, KernelTransFull<FloatType, USAGE>, 4, 4> macro_vec(
      KernelTransFull<FloatType, USAGE>(), static_cast<Deduced>(ALPHA),
      static_cast<Deduced>(BETA));
  MacroTransScalar<FloatType, USAGE> macro_scalar(static_cast<Deduced>(ALPHA),
      static_cast<Deduced>(BETA));

  auto len = macro_vec.get_cont_len();
  auto input_full_vec_len = len * (input_leading / len);
  auto output_full_vec_len = len * (output_leading / len);
  double hptc_time = 0.0;
  if (len <= input_leading and len <= output_leading) {
    OpForTrans<ORDER, ParamTrans<FloatType, ORDER, USAGE>> for_loop_vec(param);
    OpForTrans<ORDER, ParamTrans<FloatType, ORDER, USAGE>> for_loop_sca(param);
    OpForTrans<ORDER, ParamTrans<FloatType, ORDER, USAGE>> for_loop_sca_end(
        param);
    for_loop_vec.next = &for_loop_sca;
    for_loop_sca.next = &for_loop_sca_end;

    for (TensorOrder idx = 0; idx < ORDER; ++idx) {
      for_loop_vec.set_end(param->input_tensor.get_size()[idx], idx);
      for_loop_sca.set_end(param->input_tensor.get_size()[idx], idx);
      for_loop_sca_end.set_end(param->input_tensor.get_size()[idx], idx);

      if (0 == idx) {
        for_loop_vec.set_step(len, idx);
        for_loop_vec.set_end(input_full_vec_len - 1, idx);
        for_loop_sca.set_begin(input_full_vec_len, idx);
      }
      else if (perm[0] == idx) {
        for_loop_vec.set_step(len, idx);
        for_loop_vec.set_end(output_full_vec_len - 1, idx);
        for_loop_sca.set_end(output_full_vec_len, idx);
        for_loop_sca_end.set_begin(output_full_vec_len, idx);
      }
    }

    auto merge_vec = [&] (decltype(macro_vec) &v, decltype(macro_scalar) &s) {
      auto next = for_loop_vec.next;
      auto end = next->next;

      for_loop_vec(v);
      (*next)(s);
      (*end)(s);
    };

    hptc_time = timer(merge_vec, macro_vec, macro_scalar);
  }
  else {
    OpForTrans<ORDER, ParamTrans<FloatType, ORDER, USAGE>> for_loop(param);
    for (TensorOrder idx = 0; idx < ORDER; ++idx)
      for_loop.set_end(param->input_tensor.get_size()[idx], idx);
    hptc_time = timer(for_loop, macro_scalar);
  }

  // Verify results
  auto verify = data_wrapper.verify();

  // Print log
  std::stringstream ss;
  ss << "||   " << ORDER << "   || (" << perm[0];
  for (TensorOrder idx = 1; idx < ORDER; ++idx)
    ss << ", " << perm[idx];
  ss << ") || (" << param->input_tensor.get_size()[0];
  for (TensorOrder idx = 1; idx < ORDER; ++idx)
    ss << ", " << param->input_tensor.get_size()[idx];
  ss << ") || " << test_case.thread_num << " || 1 || " << std::setprecision(3)
      << ttc_time << "ms || " << std::setprecision(3) << hptc_time << "ms || "
      << (-1 == verify ? "SUCCEED! ||" : "FAILED ||");
  std::cout << ss.str() << std::endl;
}

#endif // HPTC_PERF_TEST_TEST_PERF_UTIL_TCC_
