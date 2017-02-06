#pragma once
#ifndef HPTC_CGRAPH_CGRAPH_TRANS_H_
#define HPTC_CGRAPH_CGRAPH_TRANS_H_

#include <vector>
#include <memory>
#include <thread>
#include <numeric>
#include <functional>

#include <iostream>

#include <hptc/types.h>
#include <hptc/operations/operation_trans.h>


namespace hptc {

template <typename ParamType,
          TensorOrder ORDER>
class CGraphTrans {
public:
  CGraphTrans(const std::shared_ptr<ParamType> &param,
      const std::vector<TensorOrder> &loop_order,
      const std::vector<GenNumType> &para_strategy);

  /*CGraphTrans(const CGraphTrans &graph);
  CGraphTrans<ParamType, ORDER> &operator=(const CGraphTrans &graph);*/

  ~CGraphTrans();

  INLINE void operator()();

protected:
  using For_ = OpForTrans<ParamType, ORDER>;

  void init_operations_();
  void release_operations_();

  void init_vectorize_();
  bool init_kernel_(For_ *oper, GenNumType cont_len, GenNumType ncont_len,
      TensorOrder &cont_rest, TensorOrder &ncont_rest, TensorIdx &cont_begin,
      TensorIdx &ncont_begin);
  void init_loop_order_(const std::vector<TensorOrder> &order);
  void init_parallel_(const std::vector<TensorOrder> &loop_order,
      const std::vector<GenNumType> &para_strategy);

  INLINE void task_(TensorOrder idx);

  std::shared_ptr<ParamType> param_;
  GenNumType para_strategy_, threads_;
  std::thread *thread_pool_;
  For_ *operations_;
};



/*
 * Import implementation
 */
#include "cgraph_trans.tcc"

}

#endif // HPTC_CGRAPH_CGRAPH_TRANS_H_
