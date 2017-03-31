#include <hptc/impl/hptc_trans_impl.h>

#include <vector>

#include <hptc/types.h>


hptc::CGraphTransPackBase<hptc::DoubleComplex> *create_trans_plan_impl_z(
    const hptc::DoubleComplex *in_data, hptc::DoubleComplex *out_data,
    const hptc::TensorUInt order, const std::vector<hptc::TensorUInt> &in_size,
    const std::vector<hptc::TensorUInt> &perm,
    const hptc::DeducedFloatType<hptc::DoubleComplex> alpha,
    const hptc::DeducedFloatType<hptc::DoubleComplex> beta,
    const hptc::TensorUInt num_threads, const double tuning_timeout,
    const std::vector<hptc::TensorUInt> &in_outer_size,
    const std::vector<hptc::TensorUInt> &out_outer_size) {
  return hptc::create_trans_plan_impl<hptc::DoubleComplex>(in_data, out_data,
      order, in_size, perm, alpha, beta, num_threads, tuning_timeout,
      in_outer_size, out_outer_size);
}


namespace hptc {

template class CGraphTransPackData<DoubleComplex>;
template class CGraphTransPack<DoubleComplex>;
template CGraphTransPackBase<DoubleComplex> *
create_trans_plan_impl<DoubleComplex>(const DoubleComplex *, DoubleComplex *,
    const TensorUInt, const std::vector<TensorUInt> &,
    const std::vector<TensorUInt> &, const DeducedFloatType<DoubleComplex>,
    const DeducedFloatType<DoubleComplex>, const TensorUInt, const double,
    const std::vector<TensorUInt> &, const std::vector<TensorUInt> &);

}
