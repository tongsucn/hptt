#pragma once
#ifndef HPTC_UTIL_TCC_
#define HPTC_UTIL_TCC_

template <TensorOrder ORDER>
LoopParam<ORDER>::LoopParam() {
  std::fill(this->loop_begin, this->loop_begin + ORDER, 0);
  std::fill(this->loop_end, this->loop_end + ORDER, 0);
  std::fill(this->loop_step, this->loop_step + ORDER, 0);
}


template <TensorOrder ORDER>
INLINE void LoopParam<ORDER>::set_pass(TensorOrder order) {
  std::fill(this->loop_begin, this->loop_begin + order, 0);
  std::fill(this->loop_end, this->loop_end + order, 1);
  std::fill(this->loop_step, this->loop_step + order, 1);
}


template <TensorOrder ORDER>
INLINE bool LoopParam<ORDER>::is_disabled() {
  return this->loop_begin[0] >= this->loop_end[0];
}


#endif // HPTC_UTIL_TCC_
