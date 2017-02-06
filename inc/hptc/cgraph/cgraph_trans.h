#pragma once
#ifndef HPTC_CGRAPH_CGRAPH_TRANS_H_
#define HPTC_CGRAPH_CGRAPH_TRANS_H_

#include <vector>
#include <memory>
#include <thread>

#include <hptc/types.h>
#include <hptc/operations/operation_trans.h>


namespace hptc {

template <typename ParamType,
          TensorOrder ORDER>
class CGraphTrans {
public:
  CGraphTrans(const std::shared_ptr<ParamType> &param,
      GenNumType threads = 0);

  /*CGraphTrans(const CGraphTrans &graph);
  CGraphTrans<ParamType, ORDER> &operator=(const CGraphTrans &graph);*/

  ~CGraphTrans();

  TensorIdx set_loop_order(const std::vector<TensorOrder> &order);
  TensorIdx set_parallel(const std::vector<GenNumType> &strategy,
      const GenNumType threads = 0);

  INLINE TensorIdx exec();

private:
  using For_ = OpForTrans<ParamType, ORDER>;

  void init_vectorize_();
  bool init_kernel_(For_ *oper, GenNumType cont_len, GenNumType ncont_len,
      TensorOrder &cont_rest, TensorOrder &ncont_rest, TensorIdx &cont_begin,
      TensorIdx &ncont_begin);
  void init_parallelize_();
  INLINE void thread_task_(TensorOrder idx);
  void release_operations_();

  std::shared_ptr<ParamType> param_;
  GenNumType threads_;
  For_ *operations_;
};



/*
 * Import implementation
 */
#include "cgraph_trans.tcc"

}

#endif // HPTC_CGRAPH_CGRAPH_TRANS_H_
