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
template <TensorUInt ORDER>
OpForTrans<ORDER>::OpForTrans()
    : next(nullptr) {
  this->init_disable_();
}


template <TensorUInt ORDER>
OpForTrans<ORDER>::OpForTrans(const LoopOrderTrans<ORDER> &loop_order,
    const LoopParamTrans<ORDER> &loops, const TensorUInt begin_order_idx,
    const std::array<TensorUInt, ORDER> &perm)
    : next(nullptr) {
  this->init(loop_order, loops, begin_order_idx, perm);
}


template <TensorUInt ORDER>
void OpForTrans<ORDER>::init(const LoopOrderTrans<ORDER> &loop_order,
    const LoopParamTrans<ORDER> &loops, const TensorUInt begin_order_idx,
    const std::array<TensorUInt, ORDER> &perm) {
  // Initialize loops
  this->init_loops_(loop_order, loops);

  // Initialize permutation array
  for (TensorUInt idx = 0; idx < begin_order_idx; ++idx)
    this->loop_perm_idx_[idx] = &this->loop_idx_[idx];

  for (TensorUInt idx = begin_order_idx; idx < ORDER; ++idx)
    this->loop_perm_idx_[idx] = &this->loop_idx_[perm[idx] + begin_order_idx];
}


template <TensorUInt ORDER>
template <typename MacroType,
          typename TensorType,
          typename RegType>
HPTC_INL void OpForTrans<ORDER>::operator()(const MacroType &macro_kernel,
    const TensorType &input_tensor, TensorType &output_tensor,
    const TensorIdx input_stride, const TensorIdx output_stride,
    const RegType &reg_alpha, const RegType &reg_beta) {
  this->unroller_(GenCounter<ORDER>(), macro_kernel, input_tensor,
      output_tensor, input_stride, output_stride, reg_alpha, reg_beta);
}


template <TensorUInt ORDER>
void OpForTrans<ORDER>::init_disable_() {
  std::fill(this->loop_idx_, this->loop_idx_ + ORDER, 0);
  std::fill(this->loop_perm_idx_, this->loop_perm_idx_ + ORDER, nullptr);
  std::fill(this->loop_begin_, this->loop_begin_ + ORDER, 0);
  std::fill(this->loop_end_, this->loop_end_ + ORDER, 0);
  std::fill(this->loop_step_, this->loop_step_ + ORDER, 0);

  for (TensorUInt order_idx = 0; order_idx < ORDER; ++order_idx)
    this->loop_order_[order_idx] = order_idx;
}


template <TensorUInt ORDER>
void OpForTrans<ORDER>::init_loops_(const LoopOrderTrans<ORDER> &loop_order,
    const LoopParamTrans<ORDER> &loop) {
  // Initialize loop order
  std::copy(loop_order.begin(), loop_order.end(), this->loop_order_);

  // Initialize loops
  std::copy(loop.loop_begin, loop.loop_begin + ORDER, this->loop_begin_);
  std::copy(loop.loop_end, loop.loop_end + ORDER, this->loop_end_);
  std::copy(loop.loop_step, loop.loop_step + ORDER, this->loop_step_);
}


template <TensorUInt ORDER>
template <typename MacroType,
          typename TensorType,
          typename RegType,
          TensorUInt UNROLL_NUM>
HPTC_INL void OpForTrans<ORDER>::unroller_(GenCounter<UNROLL_NUM>,
    const MacroType &macro_kernel, const TensorType &input_tensor,
    TensorType &output_tensor, const TensorIdx input_stride,
    const TensorIdx output_stride, const RegType &reg_alpha,
    const RegType &reg_beta) {
  auto for_idx = this->loop_order_[ORDER - UNROLL_NUM];
  for (this->loop_idx_[for_idx] = this->loop_begin_[for_idx];
      this->loop_idx_[for_idx] < this->loop_end_[for_idx];
      this->loop_idx_[for_idx] += this->loop_step_[for_idx])
    this->unroller_(GenCounter<UNROLL_NUM - 1>(), macro_kernel, input_tensor,
        output_tensor, input_stride, output_stride, reg_alpha, reg_beta);
}


template <TensorUInt ORDER>
template <typename MacroType,
          typename TensorType,
          typename RegType>
HPTC_INL void OpForTrans<ORDER>::unroller_(GenCounter<0>,
    const MacroType &macro_kernel, const TensorType &input_tensor,
    TensorType &output_tensor, const TensorIdx input_stride,
    const TensorIdx output_stride, const RegType &reg_alpha,
    const RegType &reg_beta) {
  macro_kernel(&input_tensor[this->loop_idx_],
      &output_tensor[this->loop_perm_idx_], input_stride, output_stride,
      reg_alpha, reg_beta);
}


/*
 * Import explicit instantiation declaration for class OpForTrans, this file
 * should be generated by cmake script.
 */
#include <hptc/gen/operation_trans_gen.tcc>

#endif // HPTC_OPERATIONS_OPERATION_TRANS_TCC_
