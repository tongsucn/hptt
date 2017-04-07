#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gen_util.gen_types import (FloatType, FLOAT_MAP)


TARGET_PREFIX = 'hptt_trans'

class IncTarget(object):
  def __init__(self, **kwargs):
    orders = kwargs['order']
    suffix = kwargs['suffix']

    # Implementation of transpose function
    self.filename = ['%s_impl_%s' % (TARGET_PREFIX, suffix)]

    # CGraphTransPackData constructor's content
    data_constructor_content = 'cgraph_trans_ptr_2_(nullptr),'
    for order in orders[1:]:
      data_constructor_content += '''
      cgraph_trans_ptr_%d_(nullptr),''' % order
    data_constructor_content = data_constructor_content[:-1]

    # CGraphTransPackData destructor's content
    data_destructor_content = ''
    for order in orders:
      data_destructor_content += '''
    delete this->cgraph_trans_ptr_%d_;''' % order

    # CGraphTransPackData's member content
    data_member_content = ''
    for order in orders:
      data_member_content += '''
  CGraphType_<%d> *cgraph_trans_ptr_%d_;''' % (order, order)

    # CGraphTransPackData's constructor content
    constructor_content = ''
    for order in orders:
      constructor_content += '''
  if (%d == order) {
    HPTT_CGRAPH_TRANS_IMPL_GEN(%d);
    this->cgraph_trans_ptr_%d_ = plan.get_graph();
  }''' % (order, order, order)

    # CGraphTransPackData's print function content
    print_content = ''
    for order in orders:
      print_content += '''
  %sif (nullptr != this->cgraph_trans_ptr_%d_) {
    auto descriptor = this->cgraph_trans_ptr_%d_->get_descriptor();
    std::cout << "Loop order: ";
    for (auto order : descriptor.loop_order)
      std::cout << order << " ";
    std::cout << std::endl;
    std::cout << "Parallelization: ";
    for (auto th_num : descriptor.parallel_strategy)
      std::cout << th_num << " ";
    std::cout << std::endl;
  }''' % ('' if order == orders[0] else 'else ', order, order)

    # CGraphTransPackData's set data function content
    set_content = ''
    for order in orders:
      set_content += '''
  %sif (nullptr != this->cgraph_trans_ptr_%d_)
    this->cgraph_trans_ptr_%d_->reset_data(in_data, out_data);''' % (
    '' if order == orders[0] else 'else ', order, order)

    # CGraphTransPackData's set thread ID function content
    set_thread_id_content = ''
    for order in orders:
      set_thread_id_content += '''
  %sif (nullptr != this->cgraph_trans_ptr_%d_)
    this->cgraph_trans_ptr_%d_->set_thread_ids(thread_ids);''' % (
    '' if order == orders[0] else 'else ', order, order)

    # CGraphTransPackData's unset thread ID function content
    unset_thread_id_content = ''
    for order in orders:
      unset_thread_id_content += '''
  %sif (nullptr != this->cgraph_trans_ptr_%d_)
    this->cgraph_trans_ptr_%d_->unset_thread_ids();''' % (
    '' if order == orders[0] else 'else ', order, order)

    # CGraphTransPackData's execution function content
    exec_content = ''
    for order in orders:
      exec_content += '''
  %sif (nullptr != this->cgraph_trans_ptr_%d_)
    this->cgraph_trans_ptr_%d_->exec();''' % (
    '' if order == orders[0] else 'else ', order, order)


    # File content
    self.content = ['''#pragma once
#ifndef HPTT_GEN_%s_IMPL_GEN_TCC_
#define HPTT_GEN_%s_IMPL_GEN_TCC_

template <typename FloatType>
class CGraphTransPackData {
public:
  CGraphTransPackData()
    : %s {
  }

  virtual ~CGraphTransPackData() {%s
  }

  constexpr static auto MIN_ORDER = %d;
  constexpr static auto MAX_ORDER = %d;

protected:
  template <TensorUInt ORDER>
  using ParamType_ = ParamTrans<TensorWrapper<FloatType, ORDER>>;
  template <TensorUInt ORDER>
  using CGraphType_ = CGraphTrans<ParamType_<ORDER>>;
%s
};


#define HPTT_CGRAPH_TRANS_IMPL_GEN(ORDER)                                     \\
  using TensorType = TensorWrapper<FloatType, ORDER>;                         \\
  using ParamType = ParamTrans<TensorType>;                                   \\
  TensorSize<ORDER> in_size_obj(in_size_vec), out_size_obj(out_size_vec),     \\
      in_outer_size_obj(in_outer_size_vec),                                   \\
      out_outer_size_obj(out_outer_size_vec);                                 \\
  std::array<TensorUInt, ORDER> perm_arr;                                     \\
  std::copy(perm.begin(), perm.end(), perm_arr.begin());                      \\
  const TensorType in_tensor(in_size_obj, in_outer_size_obj, in_data);        \\
  TensorType out_tensor(out_size_obj, out_outer_size_obj, out_data);          \\
  PlanTrans<ParamType> plan(std::make_shared<ParamType>(in_tensor, out_tensor,\\
      perm_arr, alpha, beta), num_threads, tune_loop_num, tune_para_num,      \\
      heur_loop_num, heur_para_num, tuning_timeout_ms);


template <typename FloatType>
CGraphTransPack<FloatType>::CGraphTransPack(const FloatType *in_data,
    FloatType *out_data, const TensorUInt order,
    const std::vector<TensorUInt> &in_size, const std::vector<TensorUInt> &perm,
    const DeducedFloatType<FloatType> alpha,
    const DeducedFloatType<FloatType> beta, const TensorUInt num_threads,
    const TensorInt tune_loop_num, const TensorInt tune_para_num,
    const TensorInt heur_loop_num, const TensorInt heur_para_num,
    const double tuning_timeout_ms,
    const std::vector<TensorUInt> &in_outer_size,
    const std::vector<TensorUInt> &out_outer_size)
    : CGraphTransPackBase<FloatType>(), CGraphTransPackData<FloatType>() {
  // Create input size objects
  std::vector<TensorIdx> in_size_vec(in_size.begin(), in_size.end()),
      in_outer_size_vec(in_outer_size.begin(), in_outer_size.begin());
  if (0 == in_outer_size.size())
    in_outer_size_vec = in_size_vec;

  // Create output size objects
  std::vector<TensorIdx> out_size_vec(order),
      out_outer_size_vec(out_outer_size.begin(), out_outer_size.end());
  for (TensorUInt order_idx = 0; order_idx < order; ++order_idx)
    out_size_vec[order_idx] = in_size_vec[perm[order_idx]];
  if (0 == out_outer_size.size())
    out_outer_size_vec = out_size_vec;
%s
}


template <typename FloatType>
HPTT_INL void CGraphTransPack<FloatType>::exec() {
  this->exec_impl_();
}


template <typename FloatType>
HPTT_INL void CGraphTransPack<FloatType>::operator()() {
  this->exec_impl_();
}


template <typename FloatType>
HPTT_INL void CGraphTransPack<FloatType>::print_plan() {%s
}


template <typename FloatType>
HPTT_INL void CGraphTransPack<FloatType>::reset_data(const FloatType *in_data,
    FloatType *out_data) {%s
}


template <typename FloatType>
HPTT_INL void CGraphTransPack<FloatType>::set_thread_ids(
    const std::vector<TensorInt> &thread_ids) {%s
}


template <typename FloatType>
HPTT_INL void CGraphTransPack<FloatType>::unset_thread_ids() {%s
}


template <typename FloatType>
HPTT_INL void CGraphTransPack<FloatType>::exec_impl_() {%s
}

#endif''' % (TARGET_PREFIX.upper(), TARGET_PREFIX.upper(),
    data_constructor_content, data_destructor_content, orders[0], orders[-1],
    data_member_content, constructor_content, print_content, set_content,
    set_thread_id_content, unset_thread_id_content, exec_content)]
