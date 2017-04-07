#include <hptt/impl/hptt_trans_impl.h>

#include <vector>

#include <hptt/types.h>


hptt::CGraphTransPackBase<double> *create_trans_plan_impl_d(
    const double *in_data, double *out_data, const hptt::TensorUInt order,
    const std::vector<hptt::TensorUInt> &in_size,
    const std::vector<hptt::TensorUInt> &perm,
    const hptt::DeducedFloatType<double> alpha,
    const hptt::DeducedFloatType<double> beta,
    const hptt::TensorUInt num_threads, const double tuning_timeout,
    const std::vector<hptt::TensorUInt> &in_outer_size,
    const std::vector<hptt::TensorUInt> &out_outer_size) {
  return hptt::create_trans_plan_impl<double>(in_data, out_data, order, in_size,
      perm, alpha, beta, num_threads, tuning_timeout, in_outer_size,
      out_outer_size);
}


namespace hptt {

template class CGraphTransPack<double>;
template CGraphTransPackBase<double> *create_trans_plan_impl<double>(
    const double *, double *, const TensorUInt, const std::vector<TensorUInt> &,
    const std::vector<TensorUInt> &, const DeducedFloatType<double>,
    const DeducedFloatType<double>, const TensorUInt, const double,
    const std::vector<TensorUInt> &, const std::vector<TensorUInt> &);

}
