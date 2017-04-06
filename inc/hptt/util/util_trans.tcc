#pragma once
#ifndef HPTT_UTIL_UTIL_TRANS_TCC_
#define HPTT_UTIL_UTIL_TRANS_TCC_

/*
 * Implementation for struct LoopParamTrans
 */
template <TensorUInt ORDER>
LoopParamTrans<ORDER>::LoopParamTrans() {
  this->set_disable();
}


template <TensorUInt ORDER>
void LoopParamTrans<ORDER>::set_pass(TensorUInt order) {
  std::fill(this->loop_begin, this->loop_begin + order, 0);
  std::fill(this->loop_end, this->loop_end + order, 1);
  std::fill(this->loop_step, this->loop_step + order, 1);
}


template <TensorUInt ORDER>
void LoopParamTrans<ORDER>::set_disable() {
  std::fill(this->loop_begin, this->loop_begin + ORDER, 1);
  std::fill(this->loop_end, this->loop_end + ORDER, 0);
  std::fill(this->loop_step, this->loop_step + ORDER, 1);
}


template <TensorUInt ORDER>
bool LoopParamTrans<ORDER>::is_disabled() const {
  return this->loop_begin[0] >= this->loop_end[0];
}


/*
 * Explicit template instantiation declaration for function calc_tp_trans
 */
extern template double calc_tp_trans<float>(const std::vector<TensorIdx> &,
    double);
extern template double calc_tp_trans<double>(const std::vector<TensorIdx> &,
    double);
extern template double calc_tp_trans<FloatComplex>(
    const std::vector<TensorIdx> &, double);
extern template double calc_tp_trans<DoubleComplex>(
    const std::vector<TensorIdx> &, double);

#endif // HPTT_UTIL_UTIL_TRANS_TCC_
