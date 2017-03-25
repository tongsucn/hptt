#pragma once
#ifndef HPTC_PERF_TEST_TEST_PERF_UTIL_TCC_
#define HPTC_PERF_TEST_TEST_PERF_UTIL_TCC_

template <typename FloatType,
          typename RefFuncType,
          CoefUsageTrans USAGE,
          TensorUInt ORDER>
void compare_perf(RefFuncType &ref_func, const RefTransConfig &test_case) {
  using Deduced = DeducedFloatType<FloatType>;
  using TensorType = TensorWrapper<FloatType, ORDER>;
  using Param = ParamTrans<TensorType, USAGE>;

  // Prepare data and timer
  DataWrapper<FloatType> data_wrapper(test_case.size);
  TimerWrapper timer(1);

  // Create HPTC computational graph
  std::vector<TensorUInt> perm(ORDER);
  copy(test_case.perm.begin(), test_case.perm.end(), perm.begin());
  std::vector<TensorUInt> size_vec(test_case.size.begin(),
      test_case.size.end());

  auto graph = create_cgraph_trans<FloatType>(data_wrapper.org_in_data,
      data_wrapper.act_data, ORDER, size_vec, perm,
      static_cast<Deduced>(ALPHA), static_cast<Deduced>(BETA), 0);

  double time_ttc = DBL_MAX, time_hptc = DBL_MAX;
  for (auto times = 0; times < MEASURE_REPEAT; ++times) {
    // Measure TTC version
    data_wrapper.trash_cache();
    auto new_time_ttc = timer(ref_func, data_wrapper.org_in_data,
        data_wrapper.ref_data);
    time_ttc = new_time_ttc < time_ttc ? new_time_ttc : time_ttc;

    // Measure HPTC version
    data_wrapper.trash_cache();
    auto new_time_hptc  = timer(*graph);
    time_hptc = new_time_hptc < time_hptc ? new_time_hptc : time_hptc;
  }

  auto tp_ttc = calc_tp_trans<FloatType, USAGE>(test_case.size, time_ttc);
  auto tp_hptc = calc_tp_trans<FloatType, USAGE>(test_case.size, time_hptc);

  delete graph;
  graph = nullptr;

  auto result = data_wrapper.verify();

  // Print log
  std::stringstream ss;
  ss << std::setprecision(3) << time_ttc << "," << std::setprecision(3)
      << time_hptc << "," << std::setprecision(3) << tp_ttc << ","
      << std::setprecision(3) << tp_hptc
      << (-1 == result ? ",SUCCEED" : ",FAILED");
  std::cout << ss.str() << std::endl;
}

#endif // HPTC_PERF_TEST_TEST_PERF_UTIL_TCC_
