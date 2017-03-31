#include <hptc/impl/hptc_trans_impl.h>

#include <vector>

#include <hptc/types.h>


hptc::CGraphTransPackBase<float> *create_trans_plan_impl_s(
    const float *in_data, float *out_data, const hptc::TensorUInt order,
    const std::vector<hptc::TensorUInt> &in_size,
    const std::vector<hptc::TensorUInt> &perm,
    const hptc::DeducedFloatType<float> alpha,
    const hptc::DeducedFloatType<float> beta,
    const hptc::TensorUInt num_threads, const double tuning_timeout,
    const std::vector<hptc::TensorUInt> &in_outer_size,
    const std::vector<hptc::TensorUInt> &out_outer_size) {
  return hptc::create_trans_plan_impl<float>(in_data, out_data, order, in_size,
      perm, alpha, beta, num_threads, tuning_timeout, in_outer_size,
      out_outer_size);
}


namespace hptc {

template class CGraphTransPackData<float>;
template class CGraphTransPack<float>;
template CGraphTransPackBase<float> *create_trans_plan_impl<float>(
    const float *, float *, const TensorUInt, const std::vector<TensorUInt> &,
    const std::vector<TensorUInt> &, const DeducedFloatType<float>,
    const DeducedFloatType<float>, const TensorUInt, const double,
    const std::vector<TensorUInt> &, const std::vector<TensorUInt> &);

}
