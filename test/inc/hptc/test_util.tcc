#pragma once
#ifndef HPTC_TEST_UTIL_TCC_
#define HPTC_TEST_UTIL_TCC_

/*
 * Implementation for class DataWrapper
 */
template <typename FloatType>
DataWrapper<FloatType>::DataWrapper(const std::vector<TensorOrder> &size,
    bool randomize)
    : gen_(std::random_device()()),
      dist_(this->ele_lower_, this->ele_upper_),
      data_len_(std::accumulate(size.begin(), size.end(), 1,
          std::multiplies<TensorOrder>())) {
  // Allocate memory
  this->org_in_data = new FloatType [this->data_len_];
  this->org_out_data = new FloatType [this->data_len_];
  this->ref_data = new FloatType [this->data_len_];
  this->act_data = new FloatType [this->data_len_];

  if (randomize) {
    // Initialize content with random number
#pragma omp parallel for schedule(static)
    for (TensorIdx idx = 0; idx < this->data_len_; ++idx) {
      auto org_in_ptr = reinterpret_cast<Deduced_ *>(this->org_in_data + idx);
      auto org_out_ptr = reinterpret_cast<Deduced_ *>(this->org_out_data + idx);
      auto ref_ptr = reinterpret_cast<Deduced_ *>(this->ref_data + idx);
      auto act_ptr = reinterpret_cast<Deduced_ *>(this->act_data + idx);
      for (GenNumType in_idx = 0; in_idx < this->inner_; ++in_idx) {
        org_in_ptr[in_idx] = this->dist_(this->gen_);
        org_out_ptr[in_idx] = this->dist_(this->gen_);
        ref_ptr[in_idx] = org_out_ptr[in_idx];
        act_ptr[in_idx] = org_out_ptr[in_idx];
      }
    }
  }
  else {
    // Initialize content with loop index
#pragma omp parallel for schedule(static)
    for (TensorIdx idx = 0; idx < this->data_len_; ++idx) {
      auto org_in_ptr = reinterpret_cast<Deduced_ *>(this->org_in_data + idx);
      auto org_out_ptr = reinterpret_cast<Deduced_ *>(this->org_out_data + idx);
      auto ref_ptr = reinterpret_cast<Deduced_ *>(this->ref_data + idx);
      auto act_ptr = reinterpret_cast<Deduced_ *>(this->act_data + idx);
      for (GenNumType in_idx = 0; in_idx < this->inner_; ++in_idx) {
        org_in_ptr[in_idx] = static_cast<Deduced_>(idx);
        org_out_ptr[in_idx] = static_cast<Deduced_>(idx);
        ref_ptr[in_idx] = org_out_ptr[in_idx];
        act_ptr[in_idx] = org_out_ptr[in_idx];
      }
    }
  }
}


template <typename FloatType>
DataWrapper<FloatType>::~DataWrapper() {
  delete [] this->org_in_data;
  delete [] this->org_out_data;
  delete [] this->ref_data;
  delete [] this->act_data;
}


template <typename FloatType>
void DataWrapper<FloatType>::reset_ref() {
#pragma omp parallel for schedule(static)
  for (TensorIdx idx = 0; idx < this->data_len_; ++idx)
    this->ref_data[idx] = this->org_out_data[idx];
}


template <typename FloatType>
void DataWrapper<FloatType>::reset_act() {
#pragma omp parallel for schedule(static)
  for (TensorIdx idx = 0; idx < this->data_len_; ++idx)
    this->act_data[idx] = this->org_out_data[idx];
}


template <typename FloatType>
TensorIdx DataWrapper<FloatType>::verify(
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
TensorIdx DataWrapper<FloatType>::verify() {
  return this->verify(this->ref_data, this->act_data, this->data_len_);
}

#endif // HPTC_TEST_UTIL_TCC_
