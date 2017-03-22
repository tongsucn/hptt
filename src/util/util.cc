#include <hptc/util/util.h>

#include <vector>
#include <random>
#include <numeric>
#include <functional>
#include <algorithm>
#include <utility>
#include <unordered_map>


namespace hptc {

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


/*
 * Implementation for class TimerWrapper
 */
TimerWrapper::TimerWrapper(GenNumType times)
    : times_(times) {
}


/*
 * Implementation for function approx_prod
 */
std::vector<GenNumType> approx_prod(const std::vector<GenNumType> &integers,
    const GenNumType target) {
  // Here we assume that the elements in vector integers are in ascending order

  // Data structure definition
  using Cand = std::pair<GenNumType, std::vector<GenNumType>>;
  struct CandNode {
    CandNode() {}
    CandNode(Cand &&cand) : cand(std::move(cand)) {}
    CandNode(const Cand &cand) : cand(cand) {}
    Cand cand;
    std::vector<Cand> children;
  };

  // Create vector of parents
  std::vector<CandNode> parents;
  auto best_cand = Cand(1, { 1 });

  // Iterate over all input integers
  for (auto num : integers) {
    if (num > target)
      break;

    // Create new candidate node
    auto new_cand = CandNode(Cand(num, { num }));

    // Search among parents
    auto parent_len = static_cast<GenNumType>(parents.size());
    for (GenNumType node_idx = 0; node_idx < parent_len; ++node_idx) {
      if (num * parents[node_idx].cand.first > target) {
        if (parents.back().children.empty()) {
          if (best_cand.first <= parents.back().cand.first)
            best_cand.swap(parents.back().cand);
        }
        else {
          if (parents.back().children.empty()) {
            if (best_cand.first < parents.back().cand.first or
                (best_cand.first == parents.back().cand.first and
                 best_cand.second.size() > parents.back().cand.second.size()))
              best_cand.swap(parents.back().cand);
          }
          else {
            auto &parent_best = parents.back().children.back();
            if (best_cand.first < parent_best.first or
                (best_cand.first == parent_best.first and
                 best_cand.second.size() > parent_best.second.size()))
              best_cand.swap(parent_best);
          }
        }

        // Trim and leave loop
        parents.resize(node_idx);
        break;
      }
      else {
        // Push product of num and current parent node into new candidate
        new_cand.children.push_back(Cand(num * parents[node_idx].cand.first,
              { parents[node_idx].cand.first, num }));

        // Search among children
        auto &children = parents[node_idx].children;
        GenNumType children_len = children.size();
        for (GenNumType child_idx = 0; child_idx < children_len; ++child_idx) {
          const auto &child = children[child_idx];
          if (num * child.first > target) {
            if (best_cand.first < children.back().first or
                (best_cand.first == children.back().first and
                 best_cand.second.size() > children.back().second.size()))
              best_cand.swap(children.back());

            // Trim and leave loop
            children.resize(child_idx);
            break;
          }
          else {
            new_cand.children.push_back(Cand(num * child.first, child.second));
            new_cand.children.back().second.push_back(num);
          }
        }
      }
    }

    // Append new candidate node to parents' end
    parents.push_back(new_cand);
  }

  // Check rest parents and find the best
  for (auto &parent : parents) {
    if (parent.children.empty()) {
      if (best_cand.first < parent.cand.first or
          (best_cand.first == parent.cand.first and
           best_cand.second.size() > parent.cand.second.size()))
        best_cand.swap(parent.cand);
    }
    else {
      if (best_cand.first < parent.children.back().first or
          (best_cand.first == parent.children.back().first and
           best_cand.second.size() > parent.children.back().second.size()))
        best_cand.swap(parent.children.back());
    }
  }

  // Algorithm tends to return result with smaller length
  return best_cand.second;
}


std::unordered_map<GenNumType, GenNumType> factorize(GenNumType target) {
  // Key is a prime factor, value is its frequency
  std::unordered_map<GenNumType, GenNumType> result;
  for (GenNumType num = 2; target > 1; ++num)
    if (0 == target % num)
      for (result[num] = 0; 0 == target % num; target /= num)
        ++result[num];
  return result;
}


std::vector<GenNumType> flat_map(
    const std::unordered_map<GenNumType, GenNumType> &input_map) {
  std::vector<GenNumType> result;
  for (auto kv : input_map)
    for (auto freq = 0; freq < kv.second; ++freq)
      result.push_back(kv.first);
  return result;
}


/*
 * Explicit template instantiation for class DataWrapper
 */
template class DataWrapper<float>;
template class DataWrapper<double>;
template class DataWrapper<FloatComplex>;
template class DataWrapper<DoubleComplex>;

}
