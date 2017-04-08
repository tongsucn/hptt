#pragma once
#ifndef HPTT_IMPL_HPTT_TRANS_IMPL_H_
#define HPTT_IMPL_HPTT_TRANS_IMPL_H_

#include <array>
#include <vector>
#include <memory>
#include <algorithm>
#include <iostream>

#include <hptt/types.h>
#include <hptt/tensor.h>
#include <hptt/arch/compat.h>
#include <hptt/impl/hptt_trans.h>
#include <hptt/util/util_trans.h>
#include <hptt/plan/plan_trans.h>
#include <hptt/cgraph/cgraph_trans.h>
#include <hptt/param/parameter_trans.h>


namespace hptt {

/*
 * Forward declaration of CGraphTransPack's parent class
 */
template <typename FloatType>
class CGraphTransPackData;


/*
 * Implementation of base class CGraphTransPackBase
 */
template <typename FloatType>
class CGraphTransPack final
    : public CGraphTransPackBase<FloatType>,
      public CGraphTransPackData<FloatType> {
public:
  CGraphTransPack(const FloatType *in_data, FloatType *out_data,
      const TensorUInt order, const std::vector<TensorUInt> &in_size,
      const std::vector<TensorUInt> &perm,
      const DeducedFloatType<FloatType> alpha,
      const DeducedFloatType<FloatType> beta,
      const TensorUInt num_threads, const TensorInt tune_loop_num,
      const TensorInt tune_para_num, const TensorInt heur_loop_num,
      const TensorInt heur_para_num, const double tuning_timeout_ms,
      const std::vector<TensorUInt> &in_outer_size,
      const std::vector<TensorUInt> &out_outer_size);

  // Disable copy/move constructors and operators
  CGraphTransPack(const CGraphTransPack &) = delete;
  CGraphTransPack<FloatType> &operator=(const CGraphTransPack &) = delete;
  CGraphTransPack(CGraphTransPack &&) = delete;
  CGraphTransPack<FloatType> &operator=(CGraphTransPack &&) = delete;

  virtual void exec() final;
  virtual void operator()() final;
  virtual void print_plan() final;
  virtual void reset_data(const FloatType *in_data, FloatType *out_data) final;
  virtual void set_thread_ids(const std::vector<TensorInt> &thread_ids) final;
  virtual void unset_thread_ids() final;

private:
  template <TensorUInt ORDER,
            bool UPDATE>
  using Param_ = ParamTrans<TensorWrapper<FloatType, ORDER>, UPDATE>;

  HPTT_INL void exec_impl_();

  const bool update_;
};


template <typename DeducedFloat>
bool update_output(const DeducedFloat beta) {
  return beta >= 1e-16;
}


template <typename FloatType>
CGraphTransPackBase<FloatType> *create_trans_plan_impl(const FloatType *in_data,
    FloatType *out_data, const TensorUInt order,
    const std::vector<TensorUInt> &in_size, const std::vector<TensorUInt> &perm,
    const DeducedFloatType<FloatType> alpha,
    const DeducedFloatType<FloatType> beta,
    const TensorUInt num_threads, const double tuning_timeout,
    const std::vector<TensorUInt> &in_outer_size,
    const std::vector<TensorUInt> &out_outer_size) {
  // For now, heuristic number will be limited to 7! (loop orders) x 7!
  // (parallelization strategies) candidates.
  constexpr auto heur_num = 5040;

  // Set auto-tuning amount and convert timeout from second to millisecond.
  // 64 (loop orders) x 64 (parallelization strategies) will be tuned.
  const auto tune_num = 0.0 == tuning_timeout ? 0 : 64;
  const auto tuning_timeout_ms = tuning_timeout * 1000;

  // Create transpose computational graph package
  return new CGraphTransPack<FloatType>(in_data, out_data, order, in_size, perm,
      alpha, beta, num_threads, tune_num, tune_num, heur_num, heur_num,
      tuning_timeout_ms, in_outer_size, out_outer_size);
}


/*
 * Import explicit template instantiation declaration for function
 * create_trans_plan_impl and implementations for structs
 */
#include <hptt/gen/hptt_trans_impl_gen.tcc>


/*
 * Explicit template instantiation declaration for class CGraphTransPack
 */
extern template class CGraphTransPack<float>;
extern template class CGraphTransPack<double>;
extern template class CGraphTransPack<FloatComplex>;
extern template class CGraphTransPack<DoubleComplex>;


/*
 * Explicit template instantiation declaration for function
 * create_trans_plan_impl
 */
extern template CGraphTransPackBase<float> *create_trans_plan_impl<float>(
    const float *, float *, const TensorUInt, const std::vector<TensorUInt> &,
    const std::vector<TensorUInt> &, const DeducedFloatType<float>,
    const DeducedFloatType<float>, const TensorUInt, const double,
    const std::vector<TensorUInt> &, const std::vector<TensorUInt> &);
extern template CGraphTransPackBase<double> *create_trans_plan_impl<double>(
    const double *, double *, const TensorUInt, const std::vector<TensorUInt> &,
    const std::vector<TensorUInt> &, const DeducedFloatType<double>,
    const DeducedFloatType<double>, const TensorUInt, const double,
    const std::vector<TensorUInt> &, const std::vector<TensorUInt> &);
extern template CGraphTransPackBase<FloatComplex> *
create_trans_plan_impl<FloatComplex>(const FloatComplex *, FloatComplex *,
    const TensorUInt, const std::vector<TensorUInt> &,
    const std::vector<TensorUInt> &, const DeducedFloatType<FloatComplex>,
    const DeducedFloatType<FloatComplex>, const TensorUInt, const double,
    const std::vector<TensorUInt> &, const std::vector<TensorUInt> &);
extern template CGraphTransPackBase<DoubleComplex> *
create_trans_plan_impl<DoubleComplex>(const DoubleComplex *, DoubleComplex *,
    const TensorUInt, const std::vector<TensorUInt> &,
    const std::vector<TensorUInt> &, const DeducedFloatType<DoubleComplex>,
    const DeducedFloatType<DoubleComplex>, const TensorUInt, const double,
    const std::vector<TensorUInt> &, const std::vector<TensorUInt> &);

}


extern "C" {

hptt::CGraphTransPackBase<float> *create_trans_plan_impl_s(const float *in_data,
    float *out_data, const std::vector<hptt::TensorUInt> &in_size,
    const std::vector<hptt::TensorUInt> &perm,
    const hptt::DeducedFloatType<float> alpha,
    const hptt::DeducedFloatType<float> beta,
    const hptt::TensorUInt num_threads, const double tuning_timeout,
    const std::vector<hptt::TensorUInt> &in_outer_size,
    const std::vector<hptt::TensorUInt> &out_outer_size);


hptt::CGraphTransPackBase<double> *create_trans_plan_impl_d(
    const double *in_data, double *out_data,
    const std::vector<hptt::TensorUInt> &in_size,
    const std::vector<hptt::TensorUInt> &perm,
    const hptt::DeducedFloatType<double> alpha,
    const hptt::DeducedFloatType<double> beta,
    const hptt::TensorUInt num_threads, const double tuning_timeout,
    const std::vector<hptt::TensorUInt> &in_outer_size,
    const std::vector<hptt::TensorUInt> &out_outer_size);


hptt::CGraphTransPackBase<hptt::FloatComplex> *create_trans_plan_impl_c(
    const hptt::FloatComplex *in_data, hptt::FloatComplex *out_data,
    const std::vector<hptt::TensorUInt> &in_size,
    const std::vector<hptt::TensorUInt> &perm,
    const hptt::DeducedFloatType<hptt::FloatComplex> alpha,
    const hptt::DeducedFloatType<hptt::FloatComplex> beta,
    const hptt::TensorUInt num_threads, const double tuning_timeout,
    const std::vector<hptt::TensorUInt> &in_outer_size,
    const std::vector<hptt::TensorUInt> &out_outer_size);


hptt::CGraphTransPackBase<hptt::DoubleComplex> *create_trans_plan_impl_z(
    const hptt::DoubleComplex *in_data, hptt::DoubleComplex *out_data,
    const std::vector<hptt::TensorUInt> &in_size,
    const std::vector<hptt::TensorUInt> &perm,
    const hptt::DeducedFloatType<hptt::DoubleComplex> alpha,
    const hptt::DeducedFloatType<hptt::DoubleComplex> beta,
    const hptt::TensorUInt num_threads, const double tuning_timeout,
    const std::vector<hptt::TensorUInt> &in_outer_size,
    const std::vector<hptt::TensorUInt> &out_outer_size);

}

#endif // HPTT_IMPL_HPTT_TRANS_IMPL_H_
