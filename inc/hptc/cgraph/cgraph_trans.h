#pragma once
#ifndef HPTC_CGRAPH_CGRAPH_TRANS_H_
#define HPTC_CGRAPH_CGRAPH_TRANS_H_

#include <array>
#include <vector>
#include <memory>

#include <hptc/types.h>
#include <hptc/compat.h>
#include <hptc/util/util_trans.h>
#include <hptc/operations/operation_trans.h>
#include <hptc/param/parameter_trans.h>


namespace hptc {


/*
 * Forward declaration for friend classes of CGraphTrans
 */
template <typename ParamType>
class PlanTrans;

template <typename ParamType>
class PlanTransOptimizer;


template <typename ParamType>
class CGraphTrans {
public:
  // Friend classes
  template <typename Param>
  friend class PlanTrans;
  template <typename Param>
  friend class PlanTransOptimizer;

  static constexpr auto ORDER = ParamType::ORDER;

  struct Descriptor {
    using KernelPack = typename ParamType::KernelPack;
    Descriptor();

    LoopOrderTrans<ORDER> loop_order;
    ParaStrategyTrans<ORDER> parallel_strategy;
    std::vector<std::array<LoopParamTrans<ORDER>, KernelPack::KERNEL_NUM>>
        description;
  };

  CGraphTrans(const CGraphTrans &graph) = delete;
  CGraphTrans<ParamType> &operator=(const CGraphTrans &graph) = delete;

  ~CGraphTrans();

  HPTC_INL void exec();
  HPTC_INL void operator()();
  HPTC_INL Descriptor get_descriptor() const;

private:
  using For_ = OpForTrans<ORDER>;

  CGraphTrans(const std::shared_ptr<ParamType> &param,
      const Descriptor &descriptor);

  void init(const Descriptor &descriptor);

  void release_();

  HPTC_INL void exec_general_();
  HPTC_INL void exec_common_leading_();


  std::shared_ptr<ParamType> param_;
  TensorUInt threads_;
  Descriptor descriptor_;
  For_ *operations_;
};



/*
 * Import implementation
 */
#include "cgraph_trans.tcc"

}

#endif // HPTC_CGRAPH_CGRAPH_TRANS_H_
