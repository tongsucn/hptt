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
      dist_(DataWrapper<FloatType>::ele_lower_,
          DataWrapper<FloatType>::ele_upper_),
      data_len_(std::accumulate(size.begin(), size.end(), 1,
          std::multiplies<TensorOrder>())),
      page_size_(sysconf(_SC_PAGESIZE)) {
  // Allocate memory
  posix_memalign(reinterpret_cast<void **>(&this->org_in_data),
      this->page_size_, sizeof(FloatType) * this->data_len_);
  posix_memalign(reinterpret_cast<void **>(&this->org_out_data),
      this->page_size_, sizeof(FloatType) * this->data_len_);
  posix_memalign(reinterpret_cast<void **>(&this->ref_data), this->page_size_,
      sizeof(FloatType) * this->data_len_);
  posix_memalign(reinterpret_cast<void **>(&this->act_data), this->page_size_,
      sizeof(FloatType) * this->data_len_);
  posix_memalign(reinterpret_cast<void **>(&this->trash_[0]), this->page_size_,
      sizeof(TrashType_) * DataWrapper<FloatType>::trash_size_);
  posix_memalign(reinterpret_cast<void **>(&this->trash_[1]), this->page_size_,
      sizeof(TrashType_) * DataWrapper<FloatType>::trash_size_);

  if (randomize) {
    // Initialize content with random number
#pragma omp parallel for schedule(static)
    for (TensorIdx idx = 0; idx < this->data_len_; ++idx) {
      auto org_in_ptr = reinterpret_cast<Deduced_ *>(this->org_in_data + idx);
      auto org_out_ptr = reinterpret_cast<Deduced_ *>(this->org_out_data + idx);
      auto ref_ptr = reinterpret_cast<Deduced_ *>(this->ref_data + idx);
      auto act_ptr = reinterpret_cast<Deduced_ *>(this->act_data + idx);
      for (GenNumType in_idx = 0; in_idx < inner_; ++in_idx) {
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
      for (GenNumType in_idx = 0; in_idx < inner_; ++in_idx) {
        org_in_ptr[in_idx] = static_cast<Deduced_>(idx);
        org_out_ptr[in_idx] = static_cast<Deduced_>(idx);
        ref_ptr[in_idx] = org_out_ptr[in_idx];
        act_ptr[in_idx] = org_out_ptr[in_idx];
      }
    }
  }

  // Initialize trash array
#pragma omp parallel for schedule(static)
  for (TensorIdx idx = 0; idx < DataWrapper<FloatType>::trash_size_; ++idx)
    this->trash_[0][idx] = this->trash_[1][idx] = 1.0;
}


template <typename FloatType>
DataWrapper<FloatType>::~DataWrapper() {
  free(this->org_in_data);
  free(this->org_out_data);
  free(this->ref_data);
  free(this->act_data);
  free(this->trash_[0]);
  free(this->trash_[1]);
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
void DataWrapper<FloatType>::trash_cache() {
#pragma omp parallel for schedule(static)
  for (TensorIdx idx = 0; idx < DataWrapper<FloatType>::trash_size_; ++idx)
    this->trash_[0][idx] = DataWrapper<FloatType>::trash_calc_scale_
        * this->trash_[1][idx];
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
