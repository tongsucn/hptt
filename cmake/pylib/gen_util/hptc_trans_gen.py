#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gen_util.gen_types import (FloatType, FLOAT_MAP)


TARGET_PREFIX = 'hptc_trans'

class IncTarget(object):
  def __init__(self, **kwargs):
    orders = kwargs['order']
    dtypes = kwargs['dtype']
    suffix = kwargs['suffix']

    self.filename = ['%s_base_%s' % (TARGET_PREFIX, suffix)]

    # Constructor content
    constructor_content = ''
    for order in orders:
      constructor_content += '''
    if (%d == order) {
      HPTC_CGRAPH_TRANS_BASE_GEN(%d);
      this->cgraph_trans_ptr_%d_ = plan.get_graph();
    }
    else
      this->cgraph_trans_ptr_%d_ = nullptr;''' % (order, order, order, order)

    # Destructor content
    destructor_content = ''
    for order in orders:
      destructor_content += '''
    delete this->cgraph_trans_ptr_%d_;''' % order

    # Execution function content
    exec_content = ''
    for order in orders:
      exec_content += '''
    %sif (nullptr != this->cgraph_trans_ptr_%d_)
      this->cgraph_trans_ptr_%d_->exec();''' % (
    '' if order == orders[0] else 'else ', order, order)

    # Explicit template instantiation declaration
    extern_content = ''
    for dtype in dtypes:
      extern_content += '''
extern template class CGraphTransPackBase<%s>;
extern template class CGraphTransPack<%s>;
extern template CGraphTransPack<%s> *create_trans_plan<%s>(
    const %s *, %s *, const std::vector<TensorUInt> &,
    const std::vector<TensorUInt> &, const DeducedFloatType<%s>,
    const DeducedFloatType<%s>, const TensorUInt, const double,
    const std::vector<TensorUInt> &, const std::vector<TensorUInt> &);
''' % tuple(FLOAT_MAP[dtype].full for _ in range(8))

    # Member content
    member_content = ''
    for order in orders:
      member_content += '''
  CGraphType_<%d> *cgraph_trans_ptr_%d_;''' % (order, order)


    # Base output
    self.content = ['''#pragma once
#ifndef HPTC_GEN_%s_BASE_GEN_TCC_
#define HPTC_GEN_%s_BASE_GEN_TCC_

#define HPTC_CGRAPH_TRANS_BASE_GEN(ORDER)                                     \\
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
class CGraphTransPackBase {
public:
  CGraphTransPackBase( const FloatType *in_data, FloatType *out_data,
      const TensorUInt order, const std::vector<TensorUInt> &in_size,
      const std::vector<TensorUInt> &perm,
      const DeducedFloatType<FloatType> alpha,
      const DeducedFloatType<FloatType> beta,
      const TensorUInt num_threads, const TensorInt tune_loop_num,
      const TensorInt tune_para_num, const TensorInt heur_loop_num,
      const TensorInt heur_para_num, const double tuning_timeout_ms,
      const std::vector<TensorUInt> &in_outer_size,
      const std::vector<TensorUInt> &out_outer_size) {
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

  ~CGraphTransPackBase() {%s
  }

  constexpr static auto MIN_ORDER = %d;
  constexpr static auto MAX_ORDER = %d;

protected:
  HPTC_INL void exec_base_() {%s
  }

private:
  template <TensorUInt ORDER>
  using TensorType_ = TensorWrapper<FloatType, ORDER>;
  template <TensorUInt ORDER>
  using ParamType_ = ParamTrans<TensorType_<ORDER>>;
  template <TensorUInt ORDER>
  using CGraphType_ = CGraphTrans<ParamType_<ORDER>>;
%s
};
%s
#endif''' % (TARGET_PREFIX.upper(), TARGET_PREFIX.upper(), constructor_content,
    destructor_content, orders[0], orders[-1], exec_content, member_content,
    extern_content)]

    self.filename.append('%s_%s' % (TARGET_PREFIX, suffix))
    inst_content = ''
    for dtype in dtypes:
      inst_content += '''
template class CGraphTransPackBase<%s>;
template class CGraphTransPack<%s>;
template CGraphTransPack<%s> *create_trans_plan<%s>(
    const %s *, %s *, const std::vector<TensorUInt> &,
    const std::vector<TensorUInt> &, const DeducedFloatType<%s>,
    const DeducedFloatType<%s>, const TensorUInt, const double,
    const std::vector<TensorUInt> &, const std::vector<TensorUInt> &);
''' % tuple(FLOAT_MAP[dtype].full for _ in range(8))

    self.content.append('''#pragma once
#ifndef HPTC_GEN_%s_GEN_TCC_
#define HPTC_GEN_%s_GEN_TCC_
%s
#endif
''' % (TARGET_PREFIX.upper(), TARGET_PREFIX.upper(), inst_content))
