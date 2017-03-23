#include <hptc/util/util.h>

#include <vector>
#include <utility>
#include <unordered_map>


namespace hptc {

/*
 * Implementation for class TimerWrapper
 */
TimerWrapper::TimerWrapper(TensorUInt times)
    : times_(times) {
}


/*
 * Implementation for function approx_prod
 */
std::vector<TensorUInt> approx_prod(const std::vector<TensorUInt> &integers,
    const TensorUInt target) {
  // Here we assume that the elements in vector integers are in ascending order

  // Data structure definition
  using Cand = std::pair<TensorUInt, std::vector<TensorUInt>>;
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
    auto parent_len = static_cast<TensorUInt>(parents.size());
    for (auto node_idx = 0; node_idx < parent_len; ++node_idx) {
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
        auto children_len = children.size();
        for (auto child_idx = 0; child_idx < children_len; ++child_idx) {
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


std::unordered_map<TensorUInt, TensorUInt> factorize(TensorUInt target) {
  // Key is a prime factor, value is its frequency
  std::unordered_map<TensorUInt, TensorUInt> result;
  for (auto num = 2; target > 1; ++num)
    if (0 == target % num)
      for (result[num] = 0; 0 == target % num; target /= num)
        ++result[num];
  return result;
}


std::vector<TensorUInt> flat_map(
    const std::unordered_map<TensorUInt, TensorUInt> &input_map) {
  std::vector<TensorUInt> result;
  for (auto kv : input_map)
    for (auto freq = 0; freq < kv.second; ++freq)
      result.push_back(kv.first);
  return result;
}

}
