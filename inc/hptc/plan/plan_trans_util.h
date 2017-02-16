#pragma once
#ifndef HPTC_PLAN_PLAN_TRANS_UTIL_H_
#define HPTC_PLAN_PLAN_TRANS_UTIL_H_

#include <vector>
#include <memory>
#include <utility>
#include <numeric>
#include <algorithm>

#include <omp.h>

#include <hptc/types.h>
#include <hptc/util.h>
#include <hptc/config/config_trans.h>
#include <hptc/cgraph/cgraph_trans.h>
#include <hptc/param/parameter_trans.h>


namespace hptc {

enum PlanTypeTrans {
  PLAN_TRANS_AUTO = 0x0,
  PLAN_TRANS_LOOP = 0x1,
  PLAN_TRANS_PARA = 0x2,
  PLAN_TRANS_HEUR = 0x3
};


template <typename ParamType,
          TensorOrder ORDER>
class PlanTransVectorizer {
public:
  PlanTransVectorizer(const std::shared_ptr<ParamType> &param);

  INLINE CGraphTransDescriptor<ORDER> operator()();

private:
  void init_();
  bool init_kernels_(LoopParam<ORDER> &loop, GenNumType cont_len,
      GenNumType ncont_len, TensorOrder &cont_rest, TensorOrder &ncont_rest,
      TensorIdx &cont_begin, TensorIdx &ncont_begin);

  std::shared_ptr<ParamType> param_;
  CGraphTransDescriptor<ORDER> descriptor_;
};


template <typename ParamType,
          TensorOrder ORDER>
class PlanTransParallelizer {
public:
  PlanTransParallelizer(const std::shared_ptr<ParamType> &param);

  LoopOrder<ORDER> operator()(CGraphTransDescriptor<ORDER> &descriptor,
      GenNumType threads = 0);

private:
  struct LoopNode_ {
    TensorIdx size;
    GenNumType thread_num;
    TensorOrder org_idx;
  };

  LoopOrder<ORDER> calc_depth_(const CGraphTransDescriptor<ORDER> &des,
      std::vector<GenNumType> &strategy);
  void parallelize_(CGraphTransDescriptor<ORDER> &des,
      const LoopOrder<ORDER> &loop_order,
      const std::vector<GenNumType> &strategy);

  std::shared_ptr<ParamType> param_;
};


/*
 * Import implementation
 */
#include "plan_trans_util.tcc"

}

#endif // HPTC_PLAN_PLAN_TRANS_UTIL_H_
