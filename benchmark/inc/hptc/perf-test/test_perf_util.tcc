#pragma once
#ifndef HPTC_PERF_TEST_TEST_PERF_UTIL_TCC_
#define HPTC_PERF_TEST_TEST_PERF_UTIL_TCC_

template <typename FloatType,
          typename RefFuncType,
          CoefUsageTrans USAGE,
          TensorOrder ORDER>
void compare_perf(RefFuncType &ref_func, const RefTransConfig &test_case) {
  using Deduced = DeducedFloatType<FloatType>;
  using Param = ParamTrans<FloatType, ORDER, USAGE>;

  DataWrapper<FloatType> data_wrapper(test_case.size);
  TimerWrapper timer(50);

  // Measure TTC version
  double ttc_time = timer(ref_func, data_wrapper.org_in_data,
      data_wrapper.org_out_data);

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
      data_wrapper.org_out_data);

  // 3. Create parameter
  std::array<TensorOrder, ORDER> perm;
  copy(test_case.perm.begin(), test_case.perm.end(), perm.begin());
  auto param = std::make_shared<Param>(input_tensor, output_tensor, perm,
      static_cast<Deduced>(ALPHA), static_cast<Deduced>(BETA));

  // 4. Create plan and generate computational graph
  PlanTrans<Param, ORDER> plan(param, 1);
  auto graph = plan.get_graph();

  // Execute computational graph
  double hptc_time = timer(*graph);

  delete graph;
  graph = nullptr;

  // Print log
  std::stringstream ss;
  ss << std::setprecision(3) << ttc_time << ","
      << std::setprecision(3) << hptc_time;
  std::cout << ss.str() << std::endl;
}

#endif // HPTC_PERF_TEST_TEST_PERF_UTIL_TCC_
