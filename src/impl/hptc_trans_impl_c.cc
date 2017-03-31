#include <hptc/impl/hptc_trans_impl.h>

#include <vector>

#include <hptc/types.h>


hptc::CGraphTransPackBase<hptc::FloatComplex> *create_trans_plan_impl_c(
    const hptc::FloatComplex *in_data, hptc::FloatComplex *out_data,
    const hptc::TensorUInt order, const std::vector<hptc::TensorUInt> &in_size,
    const std::vector<hptc::TensorUInt> &perm,
    const hptc::DeducedFloatType<hptc::FloatComplex> alpha,
    const hptc::DeducedFloatType<hptc::FloatComplex> beta,
    const hptc::TensorUInt num_threads, const double tuning_timeout,
    const std::vector<hptc::TensorUInt> &in_outer_size,
    const std::vector<hptc::TensorUInt> &out_outer_size) {
  return hptc::create_trans_plan_impl<hptc::FloatComplex>(in_data, out_data,
      order, in_size, perm, alpha, beta, num_threads, tuning_timeout,
      in_outer_size, out_outer_size);
}


namespace hptc {

template class CGraphTransPackData<FloatComplex>;
template class CGraphTransPack<FloatComplex>;
template CGraphTransPackBase<FloatComplex> *
create_trans_plan_impl<FloatComplex>(const FloatComplex *, FloatComplex *,
    const TensorUInt, const std::vector<TensorUInt> &,
    const std::vector<TensorUInt> &, const DeducedFloatType<FloatComplex>,
    const DeducedFloatType<FloatComplex>, const TensorUInt, const double,
    const std::vector<TensorUInt> &, const std::vector<TensorUInt> &);

}
