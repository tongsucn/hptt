#pragma once
#ifndef HPTC_CGRAPH_CGRAPH_TRANS_H_
#define HPTC_CGRAPH_CGRAPH_TRANS_H_

#include <array>
#include <vector>
#include <memory>
#include <numeric>
#include <algorithm>
#include <functional>

#include <hptc/types.h>
#include <hptc/param/parameter_trans.h>
#include <hptc/operations/operation_trans.h>


namespace hptc {

template <typename ParamType,
          TensorOrder ORDER>
class CGraphTrans {
public:
  CGraphTrans(const std::shared_ptr<ParamType> &param,
      const std::array<TensorOrder, ORDER> &loop_order,
      const std::vector<GenNumType> &strategy);

  CGraphTrans(const CGraphTrans &graph) = delete;
  CGraphTrans<ParamType, ORDER> &operator=(const CGraphTrans &graph) = delete;

  ~CGraphTrans();

  INLINE void operator()();

protected:
  // Friend class
  template <typename ParamType,
            TensorOrder ORDER>
  class PlanTrans;

  using For_ = OpForTrans<ParamType, ORDER>;

  void init_operations_();
  void release_operations_();

  void init_vectorize_();
  bool init_kernel_(For_ *oper, GenNumType cont_len, GenNumType ncont_len,
      TensorOrder &cont_rest, TensorOrder &ncont_rest, TensorIdx &cont_begin,
      TensorIdx &ncont_begin);

  void init_loop_order_();

  void init_threads_();
  void init_parallel_();

  std::shared_ptr<ParamType> param_;
  GenNumType threads_;
  std::array<TensorOrder, ORDER> loop_order_;
  std::vector<GenNumType> strategy_;
  For_ *operations_;
};



/*
 * Import implementation
 */
#include "cgraph_trans.tcc"

}

#endif // HPTC_CGRAPH_CGRAPH_TRANS_H_
