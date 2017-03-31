#include <hptc/impl/hptc_trans_impl.h>

#include <vector>

#include <hptc/types.h>


hptc::CGraphTransPackBase<double> *create_trans_plan_impl_d(
    const double *in_data, double *out_data, const hptc::TensorUInt order,
    const std::vector<hptc::TensorUInt> &in_size,
    const std::vector<hptc::TensorUInt> &perm,
    const hptc::DeducedFloatType<double> alpha,
    const hptc::DeducedFloatType<double> beta,
    const hptc::TensorUInt num_threads, const double tuning_timeout,
    const std::vector<hptc::TensorUInt> &in_outer_size,
    const std::vector<hptc::TensorUInt> &out_outer_size) {
  return hptc::create_trans_plan_impl<double>(in_data, out_data, order, in_size,
      perm, alpha, beta, num_threads, tuning_timeout, in_outer_size,
      out_outer_size);
}


namespace hptc {

template class CGraphTransPackData<double>;
template class CGraphTransPack<double>;
template CGraphTransPackBase<double> *create_trans_plan_impl<double>(
    const double *, double *, const TensorUInt, const std::vector<TensorUInt> &,
    const std::vector<TensorUInt> &, const DeducedFloatType<double>,
    const DeducedFloatType<double>, const TensorUInt, const double,
    const std::vector<TensorUInt> &, const std::vector<TensorUInt> &);

}
