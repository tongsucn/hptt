#pragma once
#ifndef HPTC_PERF_TEST_TEST_PERF_UTIL_TCC_
#define HPTC_PERF_TEST_TEST_PERF_UTIL_TCC_

template <typename FloatType,
          typename RefFuncType,
          CoefUsageTrans USAGE,
          TensorOrder ORDER>
void compare_perf(RefFuncType &ref_func, const RefTransConfig &test_case) {
  using Deduced = DeducedFloatType<FloatType>;
  using TensorType = TensorWrapper<FloatType, ORDER>;
  using Param = ParamTrans<TensorType, USAGE>;

  // Prepare data and timer
  TestDataWrapper<FloatType> data_wrapper(test_case.size);
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
  TensorType input_tensor(input_size, data_wrapper.org_in_data);
  TensorType output_tensor(output_size, data_wrapper.act_data);

  // 3. Create parameter
  std::array<TensorOrder, ORDER> perm;
  copy(test_case.perm.begin(), test_case.perm.end(), perm.begin());
  auto param = std::make_shared<Param>(input_tensor, output_tensor, perm,
      static_cast<Deduced>(ALPHA), static_cast<Deduced>(BETA));

  // 4. Create plan and generate computational graph
  PlanTrans<Param, ORDER> plan(param);
  auto graph = plan.get_graph();

  // Execute computational graph
  double hptc_time = timer(*graph);

  delete graph;
  graph = nullptr;

  auto result = data_wrapper.verify();

  // Print log
  std::stringstream ss;
  ss << std::setprecision(3) << ttc_time << ","<< std::setprecision(3)
      << hptc_time << (-1 == result ? ",SUCCEED" : ",FAILED");
  std::cout << ss.str() << std::endl;
}

#endif // HPTC_PERF_TEST_TEST_PERF_UTIL_TCC_
