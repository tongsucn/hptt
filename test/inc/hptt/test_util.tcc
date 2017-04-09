#pragma once
#ifndef HPTT_TEST_UTIL_TCC_
#define HPTT_TEST_UTIL_TCC_

/*
 * Implementation for class DataWrapper
 */
template <typename FloatType>
DataWrapper<FloatType>::DataWrapper(const std::vector<TensorIdx> &size,
    bool randomize)
    : gen_(std::random_device()()),
      dist_(DataWrapper<FloatType>::ele_lower_,
          DataWrapper<FloatType>::ele_upper_),
      data_len_(std::accumulate(size.begin(), size.end(), 1,
          std::multiplies<TensorIdx>())),
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
      for (TensorUInt in_idx = 0; in_idx < inner_; ++in_idx) {
        org_in_ptr[in_idx] = this->dist_(this->gen_);
        org_out_ptr[in_idx] = this->dist_(this->gen_);
      }
    }
#pragma omp parallel for schedule(static)
    for (TensorIdx idx = 0; idx < this->data_len_; ++idx) {
      auto org_out_ptr = reinterpret_cast<Deduced_ *>(this->org_out_data + idx);
      auto ref_ptr = reinterpret_cast<Deduced_ *>(this->ref_data + idx);
      auto act_ptr = reinterpret_cast<Deduced_ *>(this->act_data + idx);
      for (TensorUInt in_idx = 0; in_idx < inner_; ++in_idx)
        ref_ptr[in_idx] = act_ptr[in_idx] = org_out_ptr[in_idx];
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
      for (TensorUInt in_idx = 0; in_idx < inner_; ++in_idx) {
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
TensorInt DataWrapper<FloatType>::verify(
    const FloatType *ref_data, const FloatType *act_data, TensorIdx data_len) {
  using Deduced = DeducedFloatType<FloatType>;

  constexpr auto inner = DataWrapper<FloatType>::inner_;
  for (TensorIdx idx = 0; idx < data_len; ++idx) {
    auto deduced_ref = reinterpret_cast<const Deduced *>(&ref_data[idx]);
    auto deduced_act = reinterpret_cast<const Deduced *>(&act_data[idx]);
    for (TensorUInt in_idx = 0; in_idx < inner; ++in_idx) {
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
TensorInt DataWrapper<FloatType>::verify() {
  return this->verify(this->ref_data, this->act_data, this->data_len_);
}


template <typename FloatType>
void RefTrans<FloatType>::operator()(const FloatType * RESTRICT data_in,
    FloatType * RESTRICT data_out, const std::vector<TensorIdx> &size,
    const std::vector<TensorUInt> &perm,
    const DeducedFloatType<FloatType> alpha,
    const DeducedFloatType<FloatType> beta) {
  // Get order
  const auto order = static_cast<TensorUInt>(perm.size());
  if (order <= 1 or size.size() != order)
    return;

  // Initialize stride
  std::vector<TensorUInt> stride_in_outld(order, 1);
  for (TensorUInt order_idx = 0; order_idx < order - 1; ++order_idx)
    stride_in_outld[order_idx + 1]
        = stride_in_outld[order_idx] * size[order_idx];
  const auto size_inner = size[perm[0]];

  // Combine all non-stride-one orders of output tensor into a single order for
  // maximum parallelism
  TensorIdx size_outer = 1;
  for (TensorUInt order_idx = 0; order_idx < order; ++order_idx)
    if (order_idx != perm[0])
      size_outer *= size[order_idx];

#pragma omp parallel for schedule(static)
  for (TensorIdx idx = 0; idx < size_outer; ++idx) {
    TensorIdx offset_in = 0, tmp_idx = idx;
    for (TensorUInt order_idx = 1; order_idx < order; ++order_idx) {
      auto curr_idx = tmp_idx % size[perm[order_idx]];
      tmp_idx /= size[perm[order_idx]];
      offset_in += curr_idx * stride_in_outld[perm[order_idx]];
    }

    const FloatType * RESTRICT data_in_offset_ = data_in + offset_in;
    FloatType * RESTRICT data_out_offset_ = data_out + idx * size_inner;

    const auto stride_in_inner = stride_in_outld[perm[0]];
    for (TensorIdx inner_idx = 0; inner_idx < size_inner; ++inner_idx)
      data_out_offset_[inner_idx] = beta * data_out_offset_[inner_idx]
          + alpha * data_in_offset_[inner_idx * stride_in_inner];
  }
}

#endif // HPTT_TEST_UTIL_TCC_
