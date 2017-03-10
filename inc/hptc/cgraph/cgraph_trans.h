#pragma once
#ifndef HPTC_CGRAPH_CGRAPH_TRANS_H_
#define HPTC_CGRAPH_CGRAPH_TRANS_H_

#include <array>
#include <vector>
#include <memory>

#include <hptc/types.h>
#include <hptc/config/config_trans.h>
#include <hptc/operations/operation_trans.h>


namespace hptc {

template <TensorOrder ORDER>
struct CGraphTransDescriptor {
  CGraphTransDescriptor();

  LoopOrderTrans<ORDER> loop_order;
  ParaStrategyTrans<ORDER> parallel_strategy;
  std::vector<LoopGroupTrans<ORDER>> description;
};


template <typename ParamType,
          TensorOrder ORDER>
class CGraphTrans {
public:
  CGraphTrans(const CGraphTrans &graph) = delete;
  CGraphTrans<ParamType, ORDER> &operator=(const CGraphTrans &graph) = delete;

  ~CGraphTrans();

  INLINE void operator()();

protected:
  // Friend class
  template <typename ParamType,
            TensorOrder ORDER>
  friend class PlanTrans;

  using For_ = OpForTrans<ORDER>;

  CGraphTrans(const std::shared_ptr<ParamType> &param,
      const CGraphTransDescriptor<ORDER> &descriptor);

  void init_();
  void release_operations_();

  INLINE void exec_general_();
  INLINE void exec_common_leading_();
  INLINE void exec_common_leading_noncoef_();


  std::shared_ptr<ParamType> param_;
  CGraphTransDescriptor<ORDER> descriptor_;
  GenNumType threads_;
  For_ *operations_;
};



/*
 * Import implementation
 */
#include "cgraph_trans.tcc"

}

#endif // HPTC_CGRAPH_CGRAPH_TRANS_H_
