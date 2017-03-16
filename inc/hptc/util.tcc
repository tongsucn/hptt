#pragma once
#ifndef HPTC_UTIL_TCC_
#define HPTC_UTIL_TCC_

/*
 * Implementation for class TimerWrapper
 */
template <typename Callable,
          typename... Args>
INLINE double TimerWrapper::operator()(Callable &target, Args&&... args) {
  using Duration = std::chrono::duration<double, std::milli>;

  if (0 == this->times_)
    return 0.0;

  auto start = std::chrono::high_resolution_clock::now();
  target(std::forward<Args>(args)...);
  auto diff = std::chrono::high_resolution_clock::now() - start;
  auto result = std::chrono::duration_cast<Duration>(diff);

  for (GenNumType idx = 1; idx < this->times_; ++idx) {
    start = std::chrono::high_resolution_clock::now();
    target(std::forward<Args>(args)...);
    diff = std::chrono::high_resolution_clock::now() - start;
    auto duration = std::chrono::duration_cast<Duration>(diff);
    if (duration < result)
      result = duration;
  }

  return result.count();
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


/*
 * Avoid template instantiation for class DataWrapper
 */
extern template class DataWrapper<float>;
extern template class DataWrapper<double>;
extern template class DataWrapper<FloatComplex>;
extern template class DataWrapper<DoubleComplex>;

#endif // HPTC_UTIL_TCC_
