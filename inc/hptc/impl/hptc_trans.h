#pragma once
#ifndef HPTC_IMPL_HPTC_TRANS_H_
#define HPTC_IMPL_HPTC_TRANS_H_

#include <cstdint>

#include <vector>


namespace hptc {

template <typename FloatType>
class CGraphTransPackBase {
public:
  CGraphTransPackBase() = default;

  // Copy and move are disabled
  CGraphTransPackBase(const CGraphTransPackBase &) = delete;
  CGraphTransPackBase<FloatType> &
      operator=(const CGraphTransPackBase &) = delete;
  CGraphTransPackBase(CGraphTransPackBase &&) = delete;
  CGraphTransPackBase<FloatType> &operator=(CGraphTransPackBase &&) = delete;

  virtual ~CGraphTransPackBase() = default;

  virtual void exec() = 0;
  virtual void operator()() = 0;
  virtual void print_plan() = 0;
  virtual void reset_data(const FloatType *in_data, FloatType *out_data) = 0;
  virtual void set_thread_ids(const std::vector<int32_t> &thread_ids) = 0;
  virtual void unset_thread_ids() = 0;
};


template <typename FloatType>
using trans_plan = CGraphTransPackBase<FloatType>;

}

#endif // HPTC_IMPL_HPTC_TRANS_H_
