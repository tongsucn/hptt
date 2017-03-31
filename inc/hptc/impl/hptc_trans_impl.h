#pragma once
#ifndef HPTC_IMPL_HPTC_TRANS_IMPL_H_
#define HPTC_IMPL_HPTC_TRANS_IMPL_H_

#include <array>
#include <vector>
#include <memory>
#include <algorithm>

#include <hptc/types.h>
#include <hptc/tensor.h>
#include <hptc/arch/compat.h>
#include <hptc/impl/hptc_trans.h>
#include <hptc/util/util_trans.h>
#include <hptc/plan/plan_trans.h>
#include <hptc/cgraph/cgraph_trans.h>
#include <hptc/param/parameter_trans.h>


namespace hptc {

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

private:
  HPTC_INL void exec_impl_();
};


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
#include <hptc/gen/hptc_trans_impl_gen.tcc>

}


extern "C" {

hptc::CGraphTransPackBase<float> *create_trans_plan_impl_s(const float *in_data,
    float *out_data, const std::vector<hptc::TensorUInt> &in_size,
    const std::vector<hptc::TensorUInt> &perm,
    const hptc::DeducedFloatType<float> alpha,
    const hptc::DeducedFloatType<float> beta,
    const hptc::TensorUInt num_threads, const double tuning_timeout,
    const std::vector<hptc::TensorUInt> &in_outer_size,
    const std::vector<hptc::TensorUInt> &out_outer_size);


hptc::CGraphTransPackBase<double> *create_trans_plan_impl_d(
    const double *in_data, double *out_data,
    const std::vector<hptc::TensorUInt> &in_size,
    const std::vector<hptc::TensorUInt> &perm,
    const hptc::DeducedFloatType<double> alpha,
    const hptc::DeducedFloatType<double> beta,
    const hptc::TensorUInt num_threads, const double tuning_timeout,
    const std::vector<hptc::TensorUInt> &in_outer_size,
    const std::vector<hptc::TensorUInt> &out_outer_size);


hptc::CGraphTransPackBase<hptc::FloatComplex> *create_trans_plan_impl_c(
    const hptc::FloatComplex *in_data, hptc::FloatComplex *out_data,
    const std::vector<hptc::TensorUInt> &in_size,
    const std::vector<hptc::TensorUInt> &perm,
    const hptc::DeducedFloatType<hptc::FloatComplex> alpha,
    const hptc::DeducedFloatType<hptc::FloatComplex> beta,
    const hptc::TensorUInt num_threads, const double tuning_timeout,
    const std::vector<hptc::TensorUInt> &in_outer_size,
    const std::vector<hptc::TensorUInt> &out_outer_size);


hptc::CGraphTransPackBase<hptc::DoubleComplex> *create_trans_plan_impl_z(
    const hptc::DoubleComplex *in_data, hptc::DoubleComplex *out_data,
    const std::vector<hptc::TensorUInt> &in_size,
    const std::vector<hptc::TensorUInt> &perm,
    const hptc::DeducedFloatType<hptc::DoubleComplex> alpha,
    const hptc::DeducedFloatType<hptc::DoubleComplex> beta,
    const hptc::TensorUInt num_threads, const double tuning_timeout,
    const std::vector<hptc::TensorUInt> &in_outer_size,
    const std::vector<hptc::TensorUInt> &out_outer_size);

}

#endif // HPTC_IMPL_HPTC_TRANS_IMPL_H_
