#include <hptt/impl/hptt_trans_impl.h>

#include <vector>

#include <hptt/types.h>


hptt::CGraphTransPackBase<hptt::FloatComplex> *create_trans_plan_impl_c(
    const hptt::FloatComplex *in_data, hptt::FloatComplex *out_data,
    const hptt::TensorUInt order, const std::vector<hptt::TensorUInt> &in_size,
    const std::vector<hptt::TensorUInt> &perm,
    const hptt::DeducedFloatType<hptt::FloatComplex> alpha,
    const hptt::DeducedFloatType<hptt::FloatComplex> beta,
    const hptt::TensorUInt num_threads, const double tuning_timeout,
    const std::vector<hptt::TensorUInt> &in_outer_size,
    const std::vector<hptt::TensorUInt> &out_outer_size) {
  return hptt::create_trans_plan_impl<hptt::FloatComplex>(in_data, out_data,
      order, in_size, perm, alpha, beta, num_threads, tuning_timeout,
      in_outer_size, out_outer_size);
}


namespace hptt {

template class CGraphTransPackData<FloatComplex>;
template class CGraphTransPack<FloatComplex>;
template CGraphTransPackBase<FloatComplex> *
create_trans_plan_impl<FloatComplex>(const FloatComplex *, FloatComplex *,
    const TensorUInt, const std::vector<TensorUInt> &,
    const std::vector<TensorUInt> &, const DeducedFloatType<FloatComplex>,
    const DeducedFloatType<FloatComplex>, const TensorUInt, const double,
    const std::vector<TensorUInt> &, const std::vector<TensorUInt> &);

}
