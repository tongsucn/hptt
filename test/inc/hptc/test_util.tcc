#pragma once
#ifndef TEST_TEST_UTIL_TCC_
#define TEST_TEST_UTIL_TCC_

template <typename FloatType>
TestDataWrapper<FloatType>::TestDataWrapper(
    const std::vector<TensorOrder> &size)
    : DataWrapper<FloatType>(size),
      ref_data(new FloatType [this->data_len_]),
      act_data(new FloatType [this->data_len_]) {
}


template <typename FloatType>
TestDataWrapper<FloatType>::~TestDataWrapper() {
  delete [] this->ref_data;
  delete [] this->act_data;
}


template <typename FloatType>
void TestDataWrapper<FloatType>::reset_ref() {
  std::copy(this->org_out_data, this->org_out_data + this->data_len_,
      this->ref_data);
}


template <typename FloatType>
void TestDataWrapper<FloatType>::reset_act() {
  std::copy(this->org_out_data, this->org_out_data + this->data_len_,
      this->act_data);
}


template <typename FloatType>
TensorIdx TestDataWrapper<FloatType>::verify(
    const FloatType *ref_data, const FloatType *act_data, TensorIdx data_len) {
  using Deduced = DeducedFloatType<FloatType>;

  constexpr auto inner = DataWrapper<FloatType>::inner_;
  for (TensorIdx idx = 0; idx < data_len; ++idx) {
    auto deduced_ref = reinterpret_cast<const Deduced *>(&ref_data[idx]);
    auto deduced_act = reinterpret_cast<const Deduced *>(&act_data[idx]);
    for (GenNumType in_idx = 0; in_idx < inner; ++in_idx) {
      double ref_abs = std::abs(static_cast<double>(deduced_ref[in_idx]));
      double act_abs = std::abs(static_cast<double>(deduced_act[in_idx]));
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


template <typename FloatType>
TensorIdx TestDataWrapper<FloatType>::verify() {
  return this->verify(this->ref_data, this->act_data, this->data_len_);
}

#endif // TEST_TEST_UTIL_TCC_
