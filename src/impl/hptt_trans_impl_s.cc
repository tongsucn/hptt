#include <hptt/impl/hptt_trans_impl.h>

#include <vector>

#include <hptt/types.h>


hptt::CGraphTransPackBase<float> *create_trans_plan_impl_s(
    const float *in_data, float *out_data, const hptt::TensorUInt order,
    const std::vector<hptt::TensorUInt> &in_size,
    const std::vector<hptt::TensorUInt> &perm,
    const hptt::DeducedFloatType<float> alpha,
    const hptt::DeducedFloatType<float> beta,
    const hptt::TensorUInt num_threads, const double tuning_timeout,
    const std::vector<hptt::TensorUInt> &in_outer_size,
    const std::vector<hptt::TensorUInt> &out_outer_size) {
  return hptt::create_trans_plan_impl<float>(in_data, out_data, order, in_size,
      perm, alpha, beta, num_threads, tuning_timeout, in_outer_size,
      out_outer_size);
}


namespace hptt {

template class CGraphTransPack<float>;
template CGraphTransPackBase<float> *create_trans_plan_impl<float>(
    const float *, float *, const TensorUInt, const std::vector<TensorUInt> &,
    const std::vector<TensorUInt> &, const DeducedFloatType<float>,
    const DeducedFloatType<float>, const TensorUInt, const double,
    const std::vector<TensorUInt> &, const std::vector<TensorUInt> &);

}
