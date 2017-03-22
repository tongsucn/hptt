#pragma once
#ifndef HPTC_UTIL_UTIL_TRANS_TCC_
#define HPTC_UTIL_UTIL_TRANS_TCC_

/*
 * Implementation for struct LoopParamTrans
 */
template <TensorOrder ORDER>
LoopParamTrans<ORDER>::LoopParamTrans() {
  this->set_disable();
}


template <TensorOrder ORDER>
INLINE void LoopParamTrans<ORDER>::set_pass(TensorOrder order) {
  std::fill(this->loop_begin, this->loop_begin + order, 0);
  std::fill(this->loop_end, this->loop_end + order, 1);
  std::fill(this->loop_step, this->loop_step + order, 1);
}


template <TensorOrder ORDER>
INLINE void LoopParamTrans<ORDER>::set_disable() {
  std::fill(this->loop_begin, this->loop_begin + ORDER, 1);
  std::fill(this->loop_end, this->loop_end + ORDER, 0);
  std::fill(this->loop_step, this->loop_step + ORDER, 1);
}


template <TensorOrder ORDER>
INLINE bool LoopParamTrans<ORDER>::is_disabled() const {
  return this->loop_begin[0] >= this->loop_end[0];
}


/*
 * Explicit template instantiation declaration for function calc_tp_trans
 */
extern template double calc_tp_trans<float, CoefUsageTrans::USE_ALPHA>(
    const std::vector<TensorOrder> &, double);
extern template double calc_tp_trans<float, CoefUsageTrans::USE_BETA>(
    const std::vector<TensorOrder> &, double);
extern template double calc_tp_trans<float, CoefUsageTrans::USE_BOTH>(
    const std::vector<TensorOrder> &, double);
extern template double calc_tp_trans<float, CoefUsageTrans::USE_NONE>(
    const std::vector<TensorOrder> &, double);
extern template double calc_tp_trans<double, CoefUsageTrans::USE_ALPHA>(
    const std::vector<TensorOrder> &, double);
extern template double calc_tp_trans<double, CoefUsageTrans::USE_BETA>(
    const std::vector<TensorOrder> &, double);
extern template double calc_tp_trans<double, CoefUsageTrans::USE_BOTH>(
    const std::vector<TensorOrder> &, double);
extern template double calc_tp_trans<double, CoefUsageTrans::USE_NONE>(
    const std::vector<TensorOrder> &, double);
extern template double calc_tp_trans<FloatComplex, CoefUsageTrans::USE_ALPHA>(
    const std::vector<TensorOrder> &, double);
extern template double calc_tp_trans<FloatComplex, CoefUsageTrans::USE_BETA>(
    const std::vector<TensorOrder> &, double);
extern template double calc_tp_trans<FloatComplex, CoefUsageTrans::USE_BOTH>(
    const std::vector<TensorOrder> &, double);
extern template double calc_tp_trans<FloatComplex, CoefUsageTrans::USE_NONE>(
    const std::vector<TensorOrder> &, double);
extern template double calc_tp_trans<DoubleComplex, CoefUsageTrans::USE_ALPHA>(
    const std::vector<TensorOrder> &, double);
extern template double calc_tp_trans<DoubleComplex, CoefUsageTrans::USE_BETA>(
    const std::vector<TensorOrder> &, double);
extern template double calc_tp_trans<DoubleComplex, CoefUsageTrans::USE_BOTH>(
    const std::vector<TensorOrder> &, double);
extern template double calc_tp_trans<DoubleComplex, CoefUsageTrans::USE_NONE>(
    const std::vector<TensorOrder> &, double);

#endif // HPTC_UTIL_UTIL_TRANS_TCC_
