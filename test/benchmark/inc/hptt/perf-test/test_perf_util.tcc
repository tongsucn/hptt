#pragma once
#ifndef HPTT_PERF_TEST_TEST_PERF_UTIL_TCC_
#define HPTT_PERF_TEST_TEST_PERF_UTIL_TCC_

template <typename FloatType,
          typename RefFuncType,
          TensorUInt ORDER>
void compare_perf(RefFuncType &ref_trans, const RefTransConfig &test_case) {
  using Deduced = DeducedFloatType<FloatType>;

  // Prepare data and timer
  DataWrapper<FloatType> data_wrapper(test_case.size);
  TimerWrapper timer(1);

  // Create HPTT computational graph
  std::vector<TensorUInt> perm(ORDER);
  copy(test_case.perm.begin(), test_case.perm.end(), perm.begin());
  std::vector<TensorUInt> size_vec(test_case.size.begin(),
      test_case.size.end());

  auto graph = create_trans_plan<FloatType>(data_wrapper.org_in_data,
      data_wrapper.act_data, size_vec, perm, static_cast<Deduced>(ALPHA),
      static_cast<Deduced>(BETA), 0);

  double time_ttc = DBL_MAX, time_hptt = DBL_MAX;
  for (auto times = 0; times < MEASURE_REPEAT; ++times) {
    // Measure TTC version
    data_wrapper.trash_cache();
    auto new_time_ttc = timer(ref_trans, data_wrapper.org_in_data,
        data_wrapper.ref_data, test_case.size, test_case.perm,
        static_cast<Deduced>(ALPHA), static_cast<Deduced>(BETA));
    time_ttc = new_time_ttc < time_ttc ? new_time_ttc : time_ttc;

    // Measure HPTT version
    data_wrapper.trash_cache();
    auto new_time_hptt  = timer(*graph);
    time_hptt = new_time_hptt < time_hptt ? new_time_hptt : time_hptt;
  }

  auto tp_ttc = calc_tp_trans<FloatType>(test_case.size, time_ttc);
  auto tp_hptt = calc_tp_trans<FloatType>(test_case.size, time_hptt);

  delete graph;
  graph = nullptr;

  auto result = data_wrapper.verify();

  // Print log
  std::stringstream ss;
  ss << std::setprecision(3) << time_ttc << "," << std::setprecision(3)
      << time_hptt << "," << std::setprecision(3) << tp_ttc << ","
      << std::setprecision(3) << tp_hptt
      << (-1 == result ? ",SUCCEED" : ",FAILED");
  std::cout << ss.str() << std::endl;
}

#endif // HPTT_PERF_TEST_TEST_PERF_UTIL_TCC_
