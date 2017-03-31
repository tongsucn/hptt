#pragma once
#ifndef HPTC_UTIL_UTIL_TCC_
#define HPTC_UTIL_UTIL_TCC_

/*
 * Implementation for class TimerWrapper
 */
template <typename Callable,
          typename... Args>
double TimerWrapper::operator()(Callable &target, Args&&... args) {
  if (0 == this->times_)
    return 0.0;
  else if (this->is_timeout())
    return -1.0;

  double result = DBL_MAX;
  for (TensorUInt idx = 0; idx < this->times_ and not this->is_timeout();
      ++idx) {
    auto start = std::chrono::high_resolution_clock::now();
    target(std::forward<Args>(args)...);
    auto duration = std::chrono::duration_cast<Duration_>(
        std::chrono::high_resolution_clock::now() - start);
    if (duration.count() < result)
      result = duration.count();
  }

  return DBL_MAX == result ? -1.0 : result;
}


/*
 * Implementation for struct ModCmp
 */
template <typename ValType>
bool ModCmp<ValType>::operator()(const ValType &first,
    const ValType &second) {
  return 0 == first % second;
}


template <typename TargetFunc>
std::vector<TensorUInt> assign_factor(
    std::unordered_map<TensorUInt, TensorUInt> &fact_map, TensorUInt &target,
    TensorUInt &accumulate, TargetFunc cmp) {
  std::vector<TensorUInt> assigned;
  for (auto &factor : fact_map) {
    while (factor.second > 0 and cmp(target, factor.first)) {
      target /= factor.first;
      accumulate *= factor.first;
      --factor.second;
      assigned.push_back(factor.first);
    }
  }
  return assigned;
}

#endif // HPTC_UTIL_UTIL_TCC_
