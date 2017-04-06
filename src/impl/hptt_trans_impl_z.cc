#include <hptt/impl/hptt_trans_impl.h>

#include <vector>

#include <hptt/types.h>


hptt::CGraphTransPackBase<hptt::DoubleComplex> *create_trans_plan_impl_z(
    const hptt::DoubleComplex *in_data, hptt::DoubleComplex *out_data,
    const hptt::TensorUInt order, const std::vector<hptt::TensorUInt> &in_size,
    const std::vector<hptt::TensorUInt> &perm,
    const hptt::DeducedFloatType<hptt::DoubleComplex> alpha,
    const hptt::DeducedFloatType<hptt::DoubleComplex> beta,
    const hptt::TensorUInt num_threads, const double tuning_timeout,
    const std::vector<hptt::TensorUInt> &in_outer_size,
    const std::vector<hptt::TensorUInt> &out_outer_size) {
  return hptt::create_trans_plan_impl<hptt::DoubleComplex>(in_data, out_data,
      order, in_size, perm, alpha, beta, num_threads, tuning_timeout,
      in_outer_size, out_outer_size);
}


namespace hptt {

template class CGraphTransPackData<DoubleComplex>;
template class CGraphTransPack<DoubleComplex>;
template CGraphTransPackBase<DoubleComplex> *
create_trans_plan_impl<DoubleComplex>(const DoubleComplex *, DoubleComplex *,
    const TensorUInt, const std::vector<TensorUInt> &,
    const std::vector<TensorUInt> &, const DeducedFloatType<DoubleComplex>,
    const DeducedFloatType<DoubleComplex>, const TensorUInt, const double,
    const std::vector<TensorUInt> &, const std::vector<TensorUInt> &);

}
