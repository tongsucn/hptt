#include <hptt/kernels/macro_kernel_trans.h>

#include <hptt/types.h>
#include <hptt/util/util.h>
#include <hptt/arch/compat.h>
#include <hptt/kernels/micro_kernel_trans.h>


namespace hptt {

/*
 * Implementation for class MacroTrans
 */
template <typename MicroKernel,
          TensorUInt SIZE_IN_INLD,
          TensorUInt SIZE_IN_OUTLD>
MacroTrans<MicroKernel, SIZE_IN_INLD, SIZE_IN_OUTLD>::MacroTrans()
    : kernel_(),
      kn_width_(MicroKernel::KN_WIDTH) {
}


template <typename MicroKernel,
          TensorUInt SIZE_IN_INLD,
          TensorUInt SIZE_IN_OUTLD>
TensorUInt
MacroTrans<MicroKernel, SIZE_IN_INLD, SIZE_IN_OUTLD>::get_cont_len() const {
  return SIZE_IN_INLD * this->kn_width_;
}


template <typename MicroKernel,
          TensorUInt SIZE_IN_INLD,
          TensorUInt SIZE_IN_OUTLD>
TensorUInt
MacroTrans<MicroKernel, SIZE_IN_INLD, SIZE_IN_OUTLD>::get_ncont_len() const {
  return SIZE_IN_OUTLD * this->kn_width_;
}


template <typename MicroKernel,
          TensorUInt SIZE_IN_INLD,
          TensorUInt SIZE_IN_OUTLD>
void MacroTrans<MicroKernel, SIZE_IN_INLD, SIZE_IN_OUTLD>::set_coef(
    const DeducedFloatType<typename MicroKernel::Float> alpha,
    const DeducedFloatType<typename MicroKernel::Float> beta) {
  this->kernel_.set_coef(alpha, beta);
}


template <typename MicroKernel,
          TensorUInt SIZE_IN_INLD,
          TensorUInt SIZE_IN_OUTLD>
void MacroTrans<MicroKernel, SIZE_IN_INLD, SIZE_IN_OUTLD>::exec(
    const typename MicroKernel::Float *data_in,
    typename MicroKernel::Float *data_out, const TensorIdx stride_in_outld,
    const TensorIdx stride_out_inld) const {
  // Prefetching
  using FloatType = typename MicroKernel::Float;
  constexpr TensorUInt NUM_IN_OUTLD = MicroKernel::KN_WIDTH * SIZE_IN_OUTLD;
  auto USE_STREAMING = not MicroKernel::UPDATE and MicroKernel::STREAM
      and hptt::check_aligned<FloatType>(NUM_IN_OUTLD, data_out);

  if (USE_STREAMING) {
    // Create aligned buffer, macro HPTC_MEM_ALIGN is defined in
    // hptt/arch/compat.h
    constexpr TensorUInt NUM_IN_INLD = MicroKernel::KN_WIDTH * SIZE_IN_INLD;
    FloatType buffer_out[NUM_IN_INLD * NUM_IN_OUTLD]
      __attribute__((aligned(64)));

    this->tile_inld_(DualCounter<SIZE_IN_INLD, SIZE_IN_OUTLD>(), data_in,
        buffer_out, stride_in_outld, stride_out_inld);

    for (TensorUInt idx_inld = 0; idx_inld < NUM_IN_INLD; ++idx_inld)
      for (TensorUInt idx_outld = 0; idx_outld < NUM_IN_OUTLD;
          idx_outld += MicroKernel::KN_WIDTH)
        MicroKernel::sstore(data_out + idx_inld * stride_out_inld + idx_outld,
            buffer_out + idx_inld * NUM_IN_OUTLD + idx_outld);
  }
  else
    this->tile_inld_(DualCounter<SIZE_IN_INLD, SIZE_IN_OUTLD>(), data_in,
        data_out, stride_in_outld, stride_out_inld);
}


template <typename MicroKernel,
          TensorUInt SIZE_IN_INLD,
          TensorUInt SIZE_IN_OUTLD>
template <TensorUInt IN_INLD,
         TensorUInt IN_OUTLD>
void MacroTrans<MicroKernel, SIZE_IN_INLD, SIZE_IN_OUTLD>::tile_inld_(
    DualCounter<IN_INLD, IN_OUTLD>, const typename MicroKernel::Float *data_in,
    typename MicroKernel::Float *data_out, const TensorIdx stride_in_outld,
    const TensorIdx stride_out_inld) const {
  this->tile_inld_(DualCounter<IN_INLD - 1, IN_OUTLD>(), data_in, data_out,
      stride_in_outld, stride_out_inld);
  this->tile_outld_(DualCounter<IN_INLD - 1, IN_OUTLD - 1>(), data_in, data_out,
      stride_in_outld, stride_out_inld);
}


template <typename MicroKernel,
          TensorUInt SIZE_IN_INLD,
          TensorUInt SIZE_IN_OUTLD>
template <TensorUInt IN_OUTLD>
void MacroTrans<MicroKernel, SIZE_IN_INLD, SIZE_IN_OUTLD>::tile_inld_(
    DualCounter<0, IN_OUTLD>, const typename MicroKernel::Float *data_in,
    typename MicroKernel::Float *data_out, const TensorIdx stride_in_outld,
    const TensorIdx stride_out_inld) const {
}


/*
 * Explicit template instantiation for class MacroTrans
 */
template class MacroTrans<KernelTransFull<float, true>, 4, 4>;
template class MacroTrans<KernelTransFull<float, true>, 4, 3>;
template class MacroTrans<KernelTransFull<float, true>, 4, 2>;
template class MacroTrans<KernelTransFull<float, true>, 4, 1>;
template class MacroTrans<KernelTransFull<float, true>, 3, 4>;
template class MacroTrans<KernelTransFull<float, true>, 3, 3>;
template class MacroTrans<KernelTransFull<float, true>, 3, 2>;
template class MacroTrans<KernelTransFull<float, true>, 3, 1>;
template class MacroTrans<KernelTransFull<float, true>, 2, 4>;
template class MacroTrans<KernelTransFull<float, true>, 2, 3>;
template class MacroTrans<KernelTransFull<float, true>, 2, 2>;
template class MacroTrans<KernelTransFull<float, true>, 2, 1>;
template class MacroTrans<KernelTransFull<float, true>, 1, 4>;
template class MacroTrans<KernelTransFull<float, true>, 1, 3>;
template class MacroTrans<KernelTransFull<float, true>, 1, 2>;
template class MacroTrans<KernelTransFull<float, true>, 1, 1>;

template class MacroTrans<KernelTransFull<double, true>, 4, 4>;
template class MacroTrans<KernelTransFull<double, true>, 4, 3>;
template class MacroTrans<KernelTransFull<double, true>, 4, 2>;
template class MacroTrans<KernelTransFull<double, true>, 4, 1>;
template class MacroTrans<KernelTransFull<double, true>, 3, 4>;
template class MacroTrans<KernelTransFull<double, true>, 3, 3>;
template class MacroTrans<KernelTransFull<double, true>, 3, 2>;
template class MacroTrans<KernelTransFull<double, true>, 3, 1>;
template class MacroTrans<KernelTransFull<double, true>, 2, 4>;
template class MacroTrans<KernelTransFull<double, true>, 2, 3>;
template class MacroTrans<KernelTransFull<double, true>, 2, 2>;
template class MacroTrans<KernelTransFull<double, true>, 2, 1>;
template class MacroTrans<KernelTransFull<double, true>, 1, 4>;
template class MacroTrans<KernelTransFull<double, true>, 1, 3>;
template class MacroTrans<KernelTransFull<double, true>, 1, 2>;
template class MacroTrans<KernelTransFull<double, true>, 1, 1>;

template class MacroTrans<KernelTransFull<FloatComplex, true>, 4, 4>;
template class MacroTrans<KernelTransFull<FloatComplex, true>, 4, 3>;
template class MacroTrans<KernelTransFull<FloatComplex, true>, 4, 2>;
template class MacroTrans<KernelTransFull<FloatComplex, true>, 4, 1>;
template class MacroTrans<KernelTransFull<FloatComplex, true>, 3, 4>;
template class MacroTrans<KernelTransFull<FloatComplex, true>, 3, 3>;
template class MacroTrans<KernelTransFull<FloatComplex, true>, 3, 2>;
template class MacroTrans<KernelTransFull<FloatComplex, true>, 3, 1>;
template class MacroTrans<KernelTransFull<FloatComplex, true>, 2, 4>;
template class MacroTrans<KernelTransFull<FloatComplex, true>, 2, 3>;
template class MacroTrans<KernelTransFull<FloatComplex, true>, 2, 2>;
template class MacroTrans<KernelTransFull<FloatComplex, true>, 2, 1>;
template class MacroTrans<KernelTransFull<FloatComplex, true>, 1, 4>;
template class MacroTrans<KernelTransFull<FloatComplex, true>, 1, 3>;
template class MacroTrans<KernelTransFull<FloatComplex, true>, 1, 2>;
template class MacroTrans<KernelTransFull<FloatComplex, true>, 1, 1>;

template class MacroTrans<KernelTransFull<DoubleComplex, true>, 4, 4>;
template class MacroTrans<KernelTransFull<DoubleComplex, true>, 4, 3>;
template class MacroTrans<KernelTransFull<DoubleComplex, true>, 4, 2>;
template class MacroTrans<KernelTransFull<DoubleComplex, true>, 4, 1>;
template class MacroTrans<KernelTransFull<DoubleComplex, true>, 3, 4>;
template class MacroTrans<KernelTransFull<DoubleComplex, true>, 3, 3>;
template class MacroTrans<KernelTransFull<DoubleComplex, true>, 3, 2>;
template class MacroTrans<KernelTransFull<DoubleComplex, true>, 3, 1>;
template class MacroTrans<KernelTransFull<DoubleComplex, true>, 2, 4>;
template class MacroTrans<KernelTransFull<DoubleComplex, true>, 2, 3>;
template class MacroTrans<KernelTransFull<DoubleComplex, true>, 2, 2>;
template class MacroTrans<KernelTransFull<DoubleComplex, true>, 2, 1>;
template class MacroTrans<KernelTransFull<DoubleComplex, true>, 1, 4>;
template class MacroTrans<KernelTransFull<DoubleComplex, true>, 1, 3>;
template class MacroTrans<KernelTransFull<DoubleComplex, true>, 1, 2>;
template class MacroTrans<KernelTransFull<DoubleComplex, true>, 1, 1>;

template class MacroTrans<KernelTransHalf<float, true>, 4, 1>;
template class MacroTrans<KernelTransHalf<float, true>, 3, 1>;
template class MacroTrans<KernelTransHalf<float, true>, 2, 1>;
template class MacroTrans<KernelTransHalf<float, true>, 1, 4>;
template class MacroTrans<KernelTransHalf<float, true>, 1, 3>;
template class MacroTrans<KernelTransHalf<float, true>, 1, 2>;
template class MacroTrans<KernelTransHalf<float, true>, 1, 1>;

template class MacroTrans<KernelTransHalf<double, true>, 4, 1>;
template class MacroTrans<KernelTransHalf<double, true>, 3, 1>;
template class MacroTrans<KernelTransHalf<double, true>, 2, 1>;
template class MacroTrans<KernelTransHalf<double, true>, 1, 4>;
template class MacroTrans<KernelTransHalf<double, true>, 1, 3>;
template class MacroTrans<KernelTransHalf<double, true>, 1, 2>;
template class MacroTrans<KernelTransHalf<double, true>, 1, 1>;

template class MacroTrans<KernelTransHalf<FloatComplex, true>, 4, 1>;
template class MacroTrans<KernelTransHalf<FloatComplex, true>, 3, 1>;
template class MacroTrans<KernelTransHalf<FloatComplex, true>, 2, 1>;
template class MacroTrans<KernelTransHalf<FloatComplex, true>, 1, 4>;
template class MacroTrans<KernelTransHalf<FloatComplex, true>, 1, 3>;
template class MacroTrans<KernelTransHalf<FloatComplex, true>, 1, 2>;
template class MacroTrans<KernelTransHalf<FloatComplex, true>, 1, 1>;

template class MacroTrans<KernelTransHalf<DoubleComplex, true>, 4, 1>;
template class MacroTrans<KernelTransHalf<DoubleComplex, true>, 3, 1>;
template class MacroTrans<KernelTransHalf<DoubleComplex, true>, 2, 1>;
template class MacroTrans<KernelTransHalf<DoubleComplex, true>, 1, 4>;
template class MacroTrans<KernelTransHalf<DoubleComplex, true>, 1, 3>;
template class MacroTrans<KernelTransHalf<DoubleComplex, true>, 1, 2>;
template class MacroTrans<KernelTransHalf<DoubleComplex, true>, 1, 1>;

template class MacroTrans<KernelTransFull<float, false>, 4, 4>;
template class MacroTrans<KernelTransFull<float, false>, 4, 3>;
template class MacroTrans<KernelTransFull<float, false>, 4, 2>;
template class MacroTrans<KernelTransFull<float, false>, 4, 1>;
template class MacroTrans<KernelTransFull<float, false>, 3, 4>;
template class MacroTrans<KernelTransFull<float, false>, 3, 3>;
template class MacroTrans<KernelTransFull<float, false>, 3, 2>;
template class MacroTrans<KernelTransFull<float, false>, 3, 1>;
template class MacroTrans<KernelTransFull<float, false>, 2, 4>;
template class MacroTrans<KernelTransFull<float, false>, 2, 3>;
template class MacroTrans<KernelTransFull<float, false>, 2, 2>;
template class MacroTrans<KernelTransFull<float, false>, 2, 1>;
template class MacroTrans<KernelTransFull<float, false>, 1, 4>;
template class MacroTrans<KernelTransFull<float, false>, 1, 3>;
template class MacroTrans<KernelTransFull<float, false>, 1, 2>;
template class MacroTrans<KernelTransFull<float, false>, 1, 1>;

template class MacroTrans<KernelTransFull<double, false>, 4, 4>;
template class MacroTrans<KernelTransFull<double, false>, 4, 3>;
template class MacroTrans<KernelTransFull<double, false>, 4, 2>;
template class MacroTrans<KernelTransFull<double, false>, 4, 1>;
template class MacroTrans<KernelTransFull<double, false>, 3, 4>;
template class MacroTrans<KernelTransFull<double, false>, 3, 3>;
template class MacroTrans<KernelTransFull<double, false>, 3, 2>;
template class MacroTrans<KernelTransFull<double, false>, 3, 1>;
template class MacroTrans<KernelTransFull<double, false>, 2, 4>;
template class MacroTrans<KernelTransFull<double, false>, 2, 3>;
template class MacroTrans<KernelTransFull<double, false>, 2, 2>;
template class MacroTrans<KernelTransFull<double, false>, 2, 1>;
template class MacroTrans<KernelTransFull<double, false>, 1, 4>;
template class MacroTrans<KernelTransFull<double, false>, 1, 3>;
template class MacroTrans<KernelTransFull<double, false>, 1, 2>;
template class MacroTrans<KernelTransFull<double, false>, 1, 1>;

template class MacroTrans<KernelTransFull<FloatComplex, false>, 4, 4>;
template class MacroTrans<KernelTransFull<FloatComplex, false>, 4, 3>;
template class MacroTrans<KernelTransFull<FloatComplex, false>, 4, 2>;
template class MacroTrans<KernelTransFull<FloatComplex, false>, 4, 1>;
template class MacroTrans<KernelTransFull<FloatComplex, false>, 3, 4>;
template class MacroTrans<KernelTransFull<FloatComplex, false>, 3, 3>;
template class MacroTrans<KernelTransFull<FloatComplex, false>, 3, 2>;
template class MacroTrans<KernelTransFull<FloatComplex, false>, 3, 1>;
template class MacroTrans<KernelTransFull<FloatComplex, false>, 2, 4>;
template class MacroTrans<KernelTransFull<FloatComplex, false>, 2, 3>;
template class MacroTrans<KernelTransFull<FloatComplex, false>, 2, 2>;
template class MacroTrans<KernelTransFull<FloatComplex, false>, 2, 1>;
template class MacroTrans<KernelTransFull<FloatComplex, false>, 1, 4>;
template class MacroTrans<KernelTransFull<FloatComplex, false>, 1, 3>;
template class MacroTrans<KernelTransFull<FloatComplex, false>, 1, 2>;
template class MacroTrans<KernelTransFull<FloatComplex, false>, 1, 1>;

template class MacroTrans<KernelTransFull<DoubleComplex, false>, 4, 4>;
template class MacroTrans<KernelTransFull<DoubleComplex, false>, 4, 3>;
template class MacroTrans<KernelTransFull<DoubleComplex, false>, 4, 2>;
template class MacroTrans<KernelTransFull<DoubleComplex, false>, 4, 1>;
template class MacroTrans<KernelTransFull<DoubleComplex, false>, 3, 4>;
template class MacroTrans<KernelTransFull<DoubleComplex, false>, 3, 3>;
template class MacroTrans<KernelTransFull<DoubleComplex, false>, 3, 2>;
template class MacroTrans<KernelTransFull<DoubleComplex, false>, 3, 1>;
template class MacroTrans<KernelTransFull<DoubleComplex, false>, 2, 4>;
template class MacroTrans<KernelTransFull<DoubleComplex, false>, 2, 3>;
template class MacroTrans<KernelTransFull<DoubleComplex, false>, 2, 2>;
template class MacroTrans<KernelTransFull<DoubleComplex, false>, 2, 1>;
template class MacroTrans<KernelTransFull<DoubleComplex, false>, 1, 4>;
template class MacroTrans<KernelTransFull<DoubleComplex, false>, 1, 3>;
template class MacroTrans<KernelTransFull<DoubleComplex, false>, 1, 2>;
template class MacroTrans<KernelTransFull<DoubleComplex, false>, 1, 1>;

template class MacroTrans<KernelTransHalf<float, false>, 4, 1>;
template class MacroTrans<KernelTransHalf<float, false>, 3, 1>;
template class MacroTrans<KernelTransHalf<float, false>, 2, 1>;
template class MacroTrans<KernelTransHalf<float, false>, 1, 4>;
template class MacroTrans<KernelTransHalf<float, false>, 1, 3>;
template class MacroTrans<KernelTransHalf<float, false>, 1, 2>;
template class MacroTrans<KernelTransHalf<float, false>, 1, 1>;

template class MacroTrans<KernelTransHalf<double, false>, 4, 1>;
template class MacroTrans<KernelTransHalf<double, false>, 3, 1>;
template class MacroTrans<KernelTransHalf<double, false>, 2, 1>;
template class MacroTrans<KernelTransHalf<double, false>, 1, 4>;
template class MacroTrans<KernelTransHalf<double, false>, 1, 3>;
template class MacroTrans<KernelTransHalf<double, false>, 1, 2>;
template class MacroTrans<KernelTransHalf<double, false>, 1, 1>;

template class MacroTrans<KernelTransHalf<FloatComplex, false>, 4, 1>;
template class MacroTrans<KernelTransHalf<FloatComplex, false>, 3, 1>;
template class MacroTrans<KernelTransHalf<FloatComplex, false>, 2, 1>;
template class MacroTrans<KernelTransHalf<FloatComplex, false>, 1, 4>;
template class MacroTrans<KernelTransHalf<FloatComplex, false>, 1, 3>;
template class MacroTrans<KernelTransHalf<FloatComplex, false>, 1, 2>;
template class MacroTrans<KernelTransHalf<FloatComplex, false>, 1, 1>;

template class MacroTrans<KernelTransHalf<DoubleComplex, false>, 4, 1>;
template class MacroTrans<KernelTransHalf<DoubleComplex, false>, 3, 1>;
template class MacroTrans<KernelTransHalf<DoubleComplex, false>, 2, 1>;
template class MacroTrans<KernelTransHalf<DoubleComplex, false>, 1, 4>;
template class MacroTrans<KernelTransHalf<DoubleComplex, false>, 1, 3>;
template class MacroTrans<KernelTransHalf<DoubleComplex, false>, 1, 2>;
template class MacroTrans<KernelTransHalf<DoubleComplex, false>, 1, 1>;


/*
 * Explicit template instantiation for class MacroTransLinear
 */
template class MacroTransLinear<float, true>;
template class MacroTransLinear<double, true>;
template class MacroTransLinear<FloatComplex, true>;
template class MacroTransLinear<DoubleComplex, true>;

template class MacroTransLinear<float, false>;
template class MacroTransLinear<double, false>;
template class MacroTransLinear<FloatComplex, false>;
template class MacroTransLinear<DoubleComplex, false>;


/*
 * Explicit template instantiation for class MacroTransScalar
 */
template class MacroTransScalar<float, true>;
template class MacroTransScalar<double, true>;
template class MacroTransScalar<FloatComplex, true>;
template class MacroTransScalar<DoubleComplex, true>;

template class MacroTransScalar<float, false>;
template class MacroTransScalar<double, false>;
template class MacroTransScalar<FloatComplex, false>;
template class MacroTransScalar<DoubleComplex, false>;

}
