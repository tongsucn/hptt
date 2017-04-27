#pragma once
#ifndef HPTT_PERF_TEST_TEST_PERF_UTIL_TCC_
#define HPTT_PERF_TEST_TEST_PERF_UTIL_TCC_

template <typename FloatType,
          typename RefFuncType,
          TensorUInt ORDER>
void compare_perf(RefFuncType &ref_trans, const RefTransConfig &test_case,
    const TensorUInt bm_idx, bool randomize) {
  using Deduced = DeducedFloatType<FloatType>;

  // Prepare data and timer
  DataWrapper<FloatType> data_wrapper(test_case.size, randomize);
  TimerWrapper timer(1);

  // Create HPTT computational graph
  std::vector<TensorUInt> perm(ORDER);
  copy(test_case.perm.begin(), test_case.perm.end(), perm.begin());
  std::vector<TensorUInt> size_vec(test_case.size.begin(),
      test_case.size.end());

  auto graph = create_plan<FloatType>(data_wrapper.org_in_data,
      data_wrapper.act_data, size_vec, perm, static_cast<Deduced>(ALPHA),
      static_cast<Deduced>(BETA), 0, -1.0);

  double time_ref = DBL_MAX, time_hptt = DBL_MAX;
  for (auto times = 0; times < MEASURE_REPEAT; ++times) {
    // Measure reference version
    data_wrapper.trash_cache();
    auto new_time_ref = timer(ref_trans, data_wrapper.org_in_data,
        data_wrapper.ref_data, test_case.size, test_case.perm,
        static_cast<Deduced>(ALPHA), static_cast<Deduced>(BETA));
    time_ref = new_time_ref < time_ref ? new_time_ref : time_ref;

    // Measure HPTT version
    data_wrapper.trash_cache();
    auto new_time_hptt  = timer(*graph);
    time_hptt = new_time_hptt < time_hptt ? new_time_hptt : time_hptt;

    // Reset data
    data_wrapper.reset_act();
    data_wrapper.reset_ref();
  }

  auto tp_ref = calc_tp_trans<FloatType>(test_case.size, time_ref);
  auto tp_hptt = calc_tp_trans<FloatType>(test_case.size, time_hptt);

  auto result = data_wrapper.verify();

  // Print log
  printf(
      "|| %02d | %6.2f ms | %6.2f ms | %4.1f GiB/s | %4.1f GiB/s | %s ||\n",
      bm_idx, time_ref, time_hptt, tp_ref, tp_hptt,
      (-1 != result ? " \x1B[31mFAILED\x1B[0m"
       : time_ref >= time_hptt ? "\x1B[32mSUCCEED\x1B[0m"
       : "\x1B[33mSUCCEED\x1B[0m"));
}


void print_title(std::string title) {
  constexpr auto WIDTH = 68;

  if (title.length() > WIDTH - 2)
    title.resize(WIDTH - 2);

  for (auto idx = 0; idx < WIDTH; ++idx)
    std::cout << '=';
  std::cout << std::endl;
  const TensorInt title_deco_len = WIDTH - title.length() - 2;
  for (auto idx = 0; idx < title_deco_len / 2; ++idx)
    std::cout << '=';
  std::cout << ' ' << title << ' ';
  for (auto idx = 0; idx < title_deco_len / 2 + title_deco_len % 2; ++idx)
    std::cout << '=';
  std::cout << std::endl;
  for (auto idx = 0; idx < WIDTH; ++idx)
    std::cout << '=';
  std::cout << std::endl;
  std::cout <<
      "= IDX == REF TIME == HPTT TIME == REF TP ==== HPTT TP === RESULT ==="
      << std::endl;
  for (auto idx = 0; idx < WIDTH; ++idx)
    std::cout << '=';
  std::cout << std::endl;
}

#endif // HPTT_PERF_TEST_TEST_PERF_UTIL_TCC_
