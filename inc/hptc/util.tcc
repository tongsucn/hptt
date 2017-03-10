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
 * Avoid template instantiation for class DataWrapper
 */
extern template class DataWrapper<float>;
extern template class DataWrapper<double>;
extern template class DataWrapper<FloatComplex>;
extern template class DataWrapper<DoubleComplex>;

#endif // HPTC_UTIL_TCC_
