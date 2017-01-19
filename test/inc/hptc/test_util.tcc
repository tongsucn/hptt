#pragma once
#ifndef TEST_TEST_UTIL_TCC_
#define TEST_TEST_UTIL_TCC_

template <typename FloatType>
DataWrapper<FloatType>::DataWrapper(const std::vector<TensorOrder> &size)
    : rd(),
      gen(this->rd()),
      dist(this->ele_lower, this->ele_upper),
      size(size),
      data_len(std::accumulate(this->size.begin(), this->size.end(), 1,
          std::multiplies<TensorOrder>())),
      org_in_data(new FloatType [this->data_len]),
      org_out_data(new FloatType [this->data_len]),
      ref_data(new FloatType [this->data_len]),
      act_data(new FloatType [this->data_len]) {
  this->init();
}


template <typename FloatType>
DataWrapper<FloatType>::~DataWrapper() {
  delete [] this->org_in_data;
  delete [] this->org_out_data;
  delete [] this->ref_data;
  delete [] this->act_data;
}


template <typename FloatType>
void DataWrapper<FloatType>::init() {
  for (TensorIdx idx = 0; idx < this->data_len; ++idx) {
    auto org_in_ptr = reinterpret_cast<Deduced *>(this->org_in_data + idx);
    auto org_out_ptr = reinterpret_cast<Deduced *>(this->org_out_data + idx);
    for (GenNumType in_idx = 0; in_idx < this->inner; ++in_idx) {
      org_in_ptr[in_idx] = this->dist(this->gen);
      org_out_ptr[in_idx] = this->dist(this->gen);
    }
  }

  std::copy(this->org_out_data, this->org_out_data + this->data_len,
      this->ref_data);
  std::copy(this->org_out_data, this->org_out_data + this->data_len,
      this->act_data);
}


template <typename FloatType>
void DataWrapper<FloatType>::reset_ref() {
  std::copy(this->org_out_data, this->org_out_data + this->data_len,
      this->ref_data);
}


template <typename FloatType>
void DataWrapper<FloatType>::reset_act() {
  std::copy(this->org_out_data, this->org_out_data + this->data_len,
      this->act_data);
}


template <typename FloatType>
TensorIdx DataWrapper<FloatType>::verify(
    const FloatType *ref_data, const FloatType *act_data, TensorIdx data_len) {
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
TensorIdx DataWrapper<FloatType>::verify() {
  return this->verify(this->ref_data, this->act_data, this->data_len);
}


TimerWrapper::TimerWrapper(GenNumType times)
    : times(times) {
}


template <typename Callable,
          typename... Args>
INLINE double TimerWrapper::operator()(Callable &target, Args&&... args) {
  if (0 == this->times)
    return 0.0;

  auto start = std::chrono::high_resolution_clock::now();
  target(std::forward<Args>(args)...);
  auto diff = std::chrono::high_resolution_clock::now() - start;
  auto result = std::chrono::duration_cast<Duration>(diff);

  for (GenNumType idx = 1; idx < this->times; ++idx) {
    start = std::chrono::high_resolution_clock::now();
    target(std::forward<Args>(args)...);
    diff = std::chrono::high_resolution_clock::now() - start;
    auto duration = std::chrono::duration_cast<Duration>(diff);
    if (duration < result)
      result = duration;
  }

  return result.count();
}


RefTransConfig::RefTransConfig(TensorOrder order, TensorOrder thread_num,
    const std::vector<TensorOrder> &perm, const std::vector<TensorOrder> &size)
    : order(order),
      thread_num(thread_num),
      perm(perm),
      size(size) {
}

#endif // TEST_TEST_UTIL_TCC_
