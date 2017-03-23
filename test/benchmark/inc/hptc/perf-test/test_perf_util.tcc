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
  DataWrapper<FloatType> data_wrapper(test_case.size);
  TimerWrapper timer(5);

  // Measure TTC version
  double ttc_time = timer(ref_func, data_wrapper.org_in_data,
      data_wrapper.ref_data);

  // Measure HPTC version
  std::array<TensorOrder, ORDER> perm;
  copy(test_case.perm.begin(), test_case.perm.end(), perm.begin());

  auto graph = create_cgraph_trans<FloatType, ORDER>(
      data_wrapper.org_in_data, data_wrapper.act_data, test_case.size, perm,
      static_cast<Deduced>(ALPHA), static_cast<Deduced>(BETA), 0);

  // Execute computational graph
  double hptc_time = timer(*graph);

  auto tp_ttc = calc_tp_trans<FloatType, USAGE>(test_case.size, ttc_time);
  auto tp_hptc = calc_tp_trans<FloatType, USAGE>(test_case.size, hptc_time);

  delete graph;
  graph = nullptr;

  auto result = data_wrapper.verify();

  // Print log
  std::stringstream ss;
  ss << std::setprecision(3) << ttc_time << "," << std::setprecision(3)
      << hptc_time << "," << std::setprecision(3) << tp_ttc << ","
      << std::setprecision(3) << tp_hptc
      << (-1 == result ? ",SUCCEED" : ",FAILED");
  std::cout << ss.str() << std::endl;
}

#endif // HPTC_PERF_TEST_TEST_PERF_UTIL_TCC_
