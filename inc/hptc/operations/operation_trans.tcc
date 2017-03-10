#pragma once
#ifndef HPTC_OPERATIONS_OPERATION_TRANS_TCC_
#define HPTC_OPERATIONS_OPERATION_TRANS_TCC_

/*
 * Specialization for class OpForTrans' invalid cases
 */
template <>
class OpForTrans<0> {
  OpForTrans() = delete;
};


template <>
class OpForTrans<1> {
  OpForTrans() = delete;
};


/*
 * Implementation for class OpForTrans
 */
template <TensorOrder ORDER>
OpForTrans<ORDER>::OpForTrans()
    : next(nullptr) {
  this->init_disable_();
}


template <TensorOrder ORDER>
OpForTrans<ORDER>::OpForTrans(const LoopOrderTrans<ORDER> &loop_order,
    const LoopParamTrans<ORDER> &loops, const TensorOrder begin_order_idx,
    const TensorOrder *perm)
    : next(nullptr) {
  this->init(loop_order, loops, begin_order_idx, perm);
}


template <TensorOrder ORDER>
void OpForTrans<ORDER>::init(const LoopOrderTrans<ORDER> &loop_order,
    const LoopParamTrans<ORDER> &loops, const TensorOrder begin_order_idx,
    const TensorOrder *perm) {
  // Initialize loops
  this->init_loops_(loop_order, loops);

  // Initialize permutation array
  for (TensorOrder idx = 0; idx < begin_order_idx; ++idx)
    this->loop_perm_idx_[idx] = &this->loop_idx_[idx];

  for (auto idx = begin_order_idx; idx < ORDER; ++idx)
    this->loop_perm_idx_[idx] = &this->loop_idx_[perm[idx] + begin_order_idx];
}


template <TensorOrder ORDER>
template <typename MacroType,
          typename TensorType>
INLINE void OpForTrans<ORDER>::operator()(MacroType &macro_kernel,
    const TensorType &input_tensor, TensorType &output_tensor,
    const TensorIdx input_stride, const TensorIdx output_stride) {
  this->unroller_(GenCounter<ORDER>(), macro_kernel, input_tensor,
      output_tensor, input_stride, output_stride);
}


template <TensorOrder ORDER>
void OpForTrans<ORDER>::init_disable_() {
  std::fill(this->loop_idx_, this->loop_idx_ + ORDER, 0);
  std::fill(this->loop_perm_idx_, this->loop_perm_idx_ + ORDER, nullptr);
  std::fill(this->loop_begin_, this->loop_begin_ + ORDER, 0);
  std::fill(this->loop_end_, this->loop_end_ + ORDER, 0);
  std::fill(this->loop_step_, this->loop_step_ + ORDER, 0);

  for (TensorOrder idx = 0; idx < ORDER; ++idx)
    this->loop_order_[idx] = idx;
}


template <TensorOrder ORDER>
void OpForTrans<ORDER>::init_loops_(const LoopOrderTrans<ORDER> &loop_order,
    const LoopParamTrans<ORDER> &loop) {
  // Initialize loop order
  std::copy(loop_order.begin(), loop_order.end(), this->loop_order_);

  // Initialize loops
  std::copy(loop.loop_begin, loop.loop_begin + ORDER, this->loop_begin_);
  std::copy(loop.loop_end, loop.loop_end + ORDER, this->loop_end_);
  std::copy(loop.loop_step, loop.loop_step + ORDER, this->loop_step_);
}


template <TensorOrder ORDER>
template <typename MacroType,
          typename TensorType,
          GenNumType UNROLL_NUM>
INLINE void OpForTrans<ORDER>::unroller_(GenCounter<UNROLL_NUM>,
    MacroType &macro_kernel, const TensorType &input_tensor,
    TensorType &output_tensor, const TensorIdx input_stride,
    const TensorIdx output_stride) {
  auto for_idx = this->loop_order_[ORDER - UNROLL_NUM];
  for (this->loop_idx_[for_idx] = this->loop_begin_[for_idx];
      this->loop_idx_[for_idx] < this->loop_end_[for_idx];
      this->loop_idx_[for_idx] += this->loop_step_[for_idx])
    this->unroller_(GenCounter<UNROLL_NUM - 1>(), macro_kernel, input_tensor,
        output_tensor, input_stride, output_stride);
}


template <TensorOrder ORDER>
template <typename MacroType,
          typename TensorType>
INLINE void OpForTrans<ORDER>::unroller_(GenCounter<0>, MacroType &macro_kernel,
    const TensorType &input_tensor, TensorType &output_tensor,
    const TensorIdx input_stride, const TensorIdx output_stride) {
  macro_kernel(&input_tensor[this->loop_idx_],
      &output_tensor[this->loop_perm_idx_], input_stride, output_stride);
}


/*
 * Avoid template instantiation for class OpForTrans
 */
extern template class OpForTrans<2>;
extern template class OpForTrans<3>;
extern template class OpForTrans<4>;
extern template class OpForTrans<5>;
extern template class OpForTrans<6>;
extern template class OpForTrans<7>;
extern template class OpForTrans<8>;

#endif // HPTC_OPERATIONS_OPERATION_TRANS_TCC_
