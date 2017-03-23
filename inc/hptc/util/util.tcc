#pragma once
#ifndef HPTC_UTIL_UTIL_TCC_
#define HPTC_UTIL_UTIL_TCC_

/*
 * Implementation for class TimerWrapper
 */
template <typename Callable,
          typename... Args>
INLINE double TimerWrapper::operator()(Callable &target, Args&&... args) {
  using Duration = std::chrono::duration<double, std::milli>;

  if (0 == this->times_)
    return 0.0;

  double result = DBL_MAX;
  for (GenNumType idx = 0; idx < this->times_; ++idx) {
    auto start = std::chrono::high_resolution_clock::now();
    target(std::forward<Args>(args)...);
    auto duration = std::chrono::duration_cast<Duration>(
        std::chrono::high_resolution_clock::now() - start);
    if (duration.count() < result)
      result = duration.count();
  }

  return result;
}


/*
 * Implementation for struct ModCmp
 */
template <typename ValType>
INLINE bool ModCmp<ValType>::operator()(const ValType &first,
    const ValType &second) {
  return 0 == first % second;
}


template <typename TargetFunc>
std::vector<GenNumType> assign_factor(
    std::unordered_map<GenNumType, GenNumType> &fact_map, GenNumType &target,
    GenNumType &accumulate, TargetFunc cmp) {
  std::vector<GenNumType> assigned;
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
