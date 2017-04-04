#pragma once
#ifndef HPTC_CGRAPH_CGRAPH_TRANS_H_
#define HPTC_CGRAPH_CGRAPH_TRANS_H_

#include <array>
#include <vector>
#include <memory>
#include <unordered_map>

#include <omp.h>

#include <hptc/types.h>
#include <hptc/arch/compat.h>
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

  using Float = typename ParamType::Float;
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
  HPTC_INL void reset_data(const Float *data_in, Float *data_out);
  HPTC_INL void set_thread_ids(const std::vector<TensorInt> &thread_ids);
  HPTC_INL void unset_thread_ids();

private:
  CGraphTrans(const std::shared_ptr<ParamType> &param,
      const Descriptor &descriptor);

  void init(const Descriptor &descriptor);

  void release_();

  HPTC_INL void exec_general_(const TensorUInt th_idx);
  HPTC_INL void exec_common_leading_(const TensorUInt th_idx);


  std::shared_ptr<ParamType> param_;
  TensorUInt threads_;
  Descriptor descriptor_;
  OpForTrans<ORDER> *operations_;
  std::unordered_map<TensorInt, TensorUInt> thread_id_map_;
};



/*
 * Import implementation
 */
#include "cgraph_trans.tcc"

}

#endif // HPTC_CGRAPH_CGRAPH_TRANS_H_
