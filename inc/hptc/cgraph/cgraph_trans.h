#pragma once
#ifndef HPTC_CGRAPH_CGRAPH_TRANS_H_
#define HPTC_CGRAPH_CGRAPH_TRANS_H_

#include <array>
#include <vector>
#include <memory>

#include <hptc/types.h>
#include <hptc/util.h>
#include <hptc/operations/operation_trans.h>


namespace hptc {

template <TensorOrder ORDER>
using CGraphTransDescriptor = std::vector<std::array<LoopParam<ORDER>, 9>>;


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

  using For_ = OpForTrans<ParamType, ORDER>;

  CGraphTrans(const std::shared_ptr<ParamType> &param,
      const LoopOrder<ORDER> loop_order,
      const CGraphTransDescriptor<ORDER> &descriptor);

  void init_(const CGraphTransDescriptor<ORDER> &descriptor);
  void release_operations_();

  std::shared_ptr<ParamType> param_;
  CGraphTransDescriptor<ORDER> descriptor;
  GenNumType threads_;
  LoopOrder<ORDER> loop_order_;
  For_ *operations_;
};



/*
 * Import implementation
 */
#include "cgraph_trans.tcc"

}

#endif // HPTC_CGRAPH_CGRAPH_TRANS_H_
