#pragma once
#ifndef TEST_TEST_UTIL_TCC_
#define TEST_TEST_UTIL_TCC_

template <typename FloatType>
TensorIdx verify(const FloatType *ref_data, const FloatType *act_data,
    TensorIdx data_len) {
  using Deduced = DeducedFloatType<FloatType>;
  constexpr TensorOrder inner_offset = sizeof(FloatType) / sizeof(Deduced);

  for (TensorIdx idx = 0; idx < data_len; ++idx) {
    auto deduced_ref = reinterpret_cast<const Deduced *>(&ref_data[idx]);
    auto deduced_act = reinterpret_cast<const Deduced *>(&act_data[idx]);
    for (TensorOrder inner_idx = 0; inner_idx < inner_offset; ++inner_idx) {
      double ref_abs = std::abs(static_cast<double>(deduced_ref[inner_idx]));
      double act_abs = std::abs(static_cast<double>(deduced_act[inner_idx]));
      double max_abs = std::max(ref_abs, act_abs);
      double diff_abs = std::abs(ref_abs - act_abs);
      if (diff_abs > 0) {
        double rel_err = diff_abs / max_abs;
        if (rel_err > 4e-5)
          return idx;
      }
    }
  }

  return -1;
}

#endif // TEST_TEST_UTIL_TCC_
