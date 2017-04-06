#pragma once
#ifndef HPTT_KERNELS_MACRO_KERNEL_TRANS_TCC_
#define HPTT_KERNELS_MACRO_KERNEL_TRANS_TCC_

/*
 * Implementation for class MacroTransData
 */
template <typename MicroKernel,
          TensorUInt SIZE_IN_INLD,
          TensorUInt SIZE_IN_OUTLD>
MacroTransData<MicroKernel, SIZE_IN_INLD, SIZE_IN_OUTLD>::MacroTransData()
    : kernel_(),
      kn_width_(MicroKernel::KN_WIDTH) {
}


template <typename MicroKernel,
          TensorUInt SIZE_IN_INLD,
          TensorUInt SIZE_IN_OUTLD>
TensorUInt
MacroTransData<MicroKernel, SIZE_IN_INLD, SIZE_IN_OUTLD>::get_cont_len() const {
  return SIZE_IN_INLD * this->kn_width_;
}


template <typename MicroKernel,
          TensorUInt SIZE_IN_INLD,
          TensorUInt SIZE_IN_OUTLD>
TensorUInt MacroTransData<MicroKernel, SIZE_IN_INLD, SIZE_IN_OUTLD>::
get_ncont_len() const {
  return SIZE_IN_OUTLD * this->kn_width_;
}


template <typename MicroKernel,
          TensorUInt SIZE_IN_INLD,
          TensorUInt SIZE_IN_OUTLD>
void MacroTransData<MicroKernel, SIZE_IN_INLD, SIZE_IN_OUTLD>::set_coef(
    const DeducedFloatType<typename MicroKernel::Float> alpha,
    const DeducedFloatType<typename MicroKernel::Float> beta) {
  this->kernel_.set_coef(alpha, beta);
}


template <typename MicroKernel,
          TensorUInt SIZE_IN_INLD,
          TensorUInt SIZE_IN_OUTLD>
void MacroTransData<MicroKernel, SIZE_IN_INLD, SIZE_IN_OUTLD>::exec(
    const typename MicroKernel::Float *data_in,
    typename MicroKernel::Float *data_out, const TensorIdx stride_in_outld,
    const TensorIdx stride_out_inld) const {
  // Tiling kernels in non-continuous direction
  this->tile_inld_(DualCounter<SIZE_IN_INLD, SIZE_IN_OUTLD>(), data_in,
      data_out, stride_in_outld, stride_out_inld);
}


template <typename MicroKernel,
          TensorUInt SIZE_IN_INLD,
          TensorUInt SIZE_IN_OUTLD>
template <TensorUInt IN_INLD,
         TensorUInt IN_OUTLD>
void MacroTransData<MicroKernel, SIZE_IN_INLD, SIZE_IN_OUTLD>::tile_inld_(
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
void MacroTransData<MicroKernel, SIZE_IN_INLD, SIZE_IN_OUTLD>::tile_inld_(
    DualCounter<0, IN_OUTLD>, const typename MicroKernel::Float *data_in,
    typename MicroKernel::Float *data_out, const TensorIdx stride_in_outld,
    const TensorIdx stride_out_inld) const {
}


/*
 * Implementation for class MacroTransData
 */
template <typename MicroKernel,
          TensorUInt SIZE_IN_INLD,
          TensorUInt SIZE_IN_OUTLD>
MacroTrans<MicroKernel, SIZE_IN_INLD, SIZE_IN_OUTLD>::MacroTrans()
    : MacroTransData<MicroKernel, SIZE_IN_INLD, SIZE_IN_OUTLD>() {
}


/*
 * Partial specialization of class MacroTrans
 */
template <typename MicroKernel>
class MacroTrans<MicroKernel, 4, 4> : public MacroTransData<MicroKernel, 4, 4> {
public:
  MacroTrans();
  void exec(const typename MicroKernel::Float *data_in,
      typename MicroKernel::Float *data_out, const TensorIdx stride_in_outld,
      const TensorIdx stride_out_inld) const;
};

template <typename MicroKernel>
class MacroTrans<MicroKernel, 4, 3> : public MacroTransData<MicroKernel, 4, 3> {
public:
  MacroTrans();
  void exec(const typename MicroKernel::Float *data_in,
      typename MicroKernel::Float *data_out, const TensorIdx stride_in_outld,
      const TensorIdx stride_out_inld) const;
};

template <typename MicroKernel>
class MacroTrans<MicroKernel, 4, 2> : public MacroTransData<MicroKernel, 4, 2> {
public:
  MacroTrans();
  void exec(const typename MicroKernel::Float *data_in,
      typename MicroKernel::Float *data_out, const TensorIdx stride_in_outld,
      const TensorIdx stride_out_inld) const;
};

template <typename MicroKernel>
class MacroTrans<MicroKernel, 4, 1> : public MacroTransData<MicroKernel, 4, 1> {
public:
  MacroTrans();
  void exec(const typename MicroKernel::Float *data_in,
      typename MicroKernel::Float *data_out, const TensorIdx stride_in_outld,
      const TensorIdx stride_out_inld) const;
};

template <typename MicroKernel>
class MacroTrans<MicroKernel, 3, 4> : public MacroTransData<MicroKernel, 3, 4> {
public:
  MacroTrans();
  void exec(const typename MicroKernel::Float *data_in,
      typename MicroKernel::Float *data_out, const TensorIdx stride_in_outld,
      const TensorIdx stride_out_inld) const;
};

template <typename MicroKernel>
class MacroTrans<MicroKernel, 3, 3> : public MacroTransData<MicroKernel, 3, 3> {
public:
  MacroTrans();
  void exec(const typename MicroKernel::Float *data_in,
      typename MicroKernel::Float *data_out, const TensorIdx stride_in_outld,
      const TensorIdx stride_out_inld) const;
};

template <typename MicroKernel>
class MacroTrans<MicroKernel, 3, 2> : public MacroTransData<MicroKernel, 3, 2> {
public:
  MacroTrans();
  void exec(const typename MicroKernel::Float *data_in,
      typename MicroKernel::Float *data_out, const TensorIdx stride_in_outld,
      const TensorIdx stride_out_inld) const;
};

template <typename MicroKernel>
class MacroTrans<MicroKernel, 3, 1> : public MacroTransData<MicroKernel, 3, 1> {
public:
  MacroTrans();
  void exec(const typename MicroKernel::Float *data_in,
      typename MicroKernel::Float *data_out, const TensorIdx stride_in_outld,
      const TensorIdx stride_out_inld) const;
};

template <typename MicroKernel>
class MacroTrans<MicroKernel, 2, 4> : public MacroTransData<MicroKernel, 2, 4> {
public:
  MacroTrans();
  void exec(const typename MicroKernel::Float *data_in,
      typename MicroKernel::Float *data_out, const TensorIdx stride_in_outld,
      const TensorIdx stride_out_inld) const;
};

template <typename MicroKernel>
class MacroTrans<MicroKernel, 2, 3> : public MacroTransData<MicroKernel, 2, 3> {
public:
  MacroTrans();
  void exec(const typename MicroKernel::Float *data_in,
      typename MicroKernel::Float *data_out, const TensorIdx stride_in_outld,
      const TensorIdx stride_out_inld) const;
};

template <typename MicroKernel>
class MacroTrans<MicroKernel, 2, 2> : public MacroTransData<MicroKernel, 2, 2> {
public:
  MacroTrans();
  void exec(const typename MicroKernel::Float *data_in,
      typename MicroKernel::Float *data_out, const TensorIdx stride_in_outld,
      const TensorIdx stride_out_inld) const;
};

template <typename MicroKernel>
class MacroTrans<MicroKernel, 2, 1> : public MacroTransData<MicroKernel, 2, 1> {
public:
  MacroTrans();
  void exec(const typename MicroKernel::Float *data_in,
      typename MicroKernel::Float *data_out, const TensorIdx stride_in_outld,
      const TensorIdx stride_out_inld) const;
};

template <typename MicroKernel>
class MacroTrans<MicroKernel, 1, 4> : public MacroTransData<MicroKernel, 1, 4> {
public:
  MacroTrans();
  void exec(const typename MicroKernel::Float *data_in,
      typename MicroKernel::Float *data_out, const TensorIdx stride_in_outld,
      const TensorIdx stride_out_inld) const;
};

template <typename MicroKernel>
class MacroTrans<MicroKernel, 1, 3> : public MacroTransData<MicroKernel, 1, 3> {
public:
  MacroTrans();
  void exec(const typename MicroKernel::Float *data_in,
      typename MicroKernel::Float *data_out, const TensorIdx stride_in_outld,
      const TensorIdx stride_out_inld) const;
};

template <typename MicroKernel>
class MacroTrans<MicroKernel, 1, 2> : public MacroTransData<MicroKernel, 1, 2> {
public:
  MacroTrans();
  void exec(const typename MicroKernel::Float *data_in,
      typename MicroKernel::Float *data_out, const TensorIdx stride_in_outld,
      const TensorIdx stride_out_inld) const;
};

template <typename MicroKernel>
class MacroTrans<MicroKernel, 1, 1> : public MacroTransData<MicroKernel, 1, 1> {
public:
  MacroTrans();
  void exec(const typename MicroKernel::Float *data_in,
      typename MicroKernel::Float *data_out, const TensorIdx stride_in_outld,
      const TensorIdx stride_out_inld) const;
};



/*
 * Explicit instantiation declaration for class MacroTrans
 */
extern template class MacroTrans<KernelTransFull<float>, 4, 4>;
extern template class MacroTrans<KernelTransFull<float>, 4, 3>;
extern template class MacroTrans<KernelTransFull<float>, 4, 2>;
extern template class MacroTrans<KernelTransFull<float>, 4, 1>;
extern template class MacroTrans<KernelTransFull<float>, 3, 4>;
extern template class MacroTrans<KernelTransFull<float>, 3, 3>;
extern template class MacroTrans<KernelTransFull<float>, 3, 2>;
extern template class MacroTrans<KernelTransFull<float>, 3, 1>;
extern template class MacroTrans<KernelTransFull<float>, 2, 4>;
extern template class MacroTrans<KernelTransFull<float>, 2, 3>;
extern template class MacroTrans<KernelTransFull<float>, 2, 2>;
extern template class MacroTrans<KernelTransFull<float>, 2, 1>;
extern template class MacroTrans<KernelTransFull<float>, 1, 4>;
extern template class MacroTrans<KernelTransFull<float>, 1, 3>;
extern template class MacroTrans<KernelTransFull<float>, 1, 2>;
extern template class MacroTrans<KernelTransFull<float>, 1, 1>;

extern template class MacroTrans<KernelTransFull<double>, 4, 4>;
extern template class MacroTrans<KernelTransFull<double>, 4, 3>;
extern template class MacroTrans<KernelTransFull<double>, 4, 2>;
extern template class MacroTrans<KernelTransFull<double>, 4, 1>;
extern template class MacroTrans<KernelTransFull<double>, 3, 4>;
extern template class MacroTrans<KernelTransFull<double>, 3, 3>;
extern template class MacroTrans<KernelTransFull<double>, 3, 2>;
extern template class MacroTrans<KernelTransFull<double>, 3, 1>;
extern template class MacroTrans<KernelTransFull<double>, 2, 4>;
extern template class MacroTrans<KernelTransFull<double>, 2, 3>;
extern template class MacroTrans<KernelTransFull<double>, 2, 2>;
extern template class MacroTrans<KernelTransFull<double>, 2, 1>;
extern template class MacroTrans<KernelTransFull<double>, 1, 4>;
extern template class MacroTrans<KernelTransFull<double>, 1, 3>;
extern template class MacroTrans<KernelTransFull<double>, 1, 2>;
extern template class MacroTrans<KernelTransFull<double>, 1, 1>;

extern template class MacroTrans<KernelTransFull<FloatComplex>, 4, 4>;
extern template class MacroTrans<KernelTransFull<FloatComplex>, 4, 3>;
extern template class MacroTrans<KernelTransFull<FloatComplex>, 4, 2>;
extern template class MacroTrans<KernelTransFull<FloatComplex>, 4, 1>;
extern template class MacroTrans<KernelTransFull<FloatComplex>, 3, 4>;
extern template class MacroTrans<KernelTransFull<FloatComplex>, 3, 3>;
extern template class MacroTrans<KernelTransFull<FloatComplex>, 3, 2>;
extern template class MacroTrans<KernelTransFull<FloatComplex>, 3, 1>;
extern template class MacroTrans<KernelTransFull<FloatComplex>, 2, 4>;
extern template class MacroTrans<KernelTransFull<FloatComplex>, 2, 3>;
extern template class MacroTrans<KernelTransFull<FloatComplex>, 2, 2>;
extern template class MacroTrans<KernelTransFull<FloatComplex>, 2, 1>;
extern template class MacroTrans<KernelTransFull<FloatComplex>, 1, 4>;
extern template class MacroTrans<KernelTransFull<FloatComplex>, 1, 3>;
extern template class MacroTrans<KernelTransFull<FloatComplex>, 1, 2>;
extern template class MacroTrans<KernelTransFull<FloatComplex>, 1, 1>;

extern template class MacroTrans<KernelTransFull<DoubleComplex>, 4, 4>;
extern template class MacroTrans<KernelTransFull<DoubleComplex>, 4, 3>;
extern template class MacroTrans<KernelTransFull<DoubleComplex>, 4, 2>;
extern template class MacroTrans<KernelTransFull<DoubleComplex>, 4, 1>;
extern template class MacroTrans<KernelTransFull<DoubleComplex>, 3, 4>;
extern template class MacroTrans<KernelTransFull<DoubleComplex>, 3, 3>;
extern template class MacroTrans<KernelTransFull<DoubleComplex>, 3, 2>;
extern template class MacroTrans<KernelTransFull<DoubleComplex>, 3, 1>;
extern template class MacroTrans<KernelTransFull<DoubleComplex>, 2, 4>;
extern template class MacroTrans<KernelTransFull<DoubleComplex>, 2, 3>;
extern template class MacroTrans<KernelTransFull<DoubleComplex>, 2, 2>;
extern template class MacroTrans<KernelTransFull<DoubleComplex>, 2, 1>;
extern template class MacroTrans<KernelTransFull<DoubleComplex>, 1, 4>;
extern template class MacroTrans<KernelTransFull<DoubleComplex>, 1, 3>;
extern template class MacroTrans<KernelTransFull<DoubleComplex>, 1, 2>;
extern template class MacroTrans<KernelTransFull<DoubleComplex>, 1, 1>;

extern template class MacroTrans<KernelTransHalf<float>, 4, 1>;
extern template class MacroTrans<KernelTransHalf<float>, 3, 1>;
extern template class MacroTrans<KernelTransHalf<float>, 2, 1>;
extern template class MacroTrans<KernelTransHalf<float>, 1, 4>;
extern template class MacroTrans<KernelTransHalf<float>, 1, 3>;
extern template class MacroTrans<KernelTransHalf<float>, 1, 2>;
extern template class MacroTrans<KernelTransHalf<float>, 1, 1>;

extern template class MacroTrans<KernelTransHalf<double>, 4, 1>;
extern template class MacroTrans<KernelTransHalf<double>, 3, 1>;
extern template class MacroTrans<KernelTransHalf<double>, 2, 1>;
extern template class MacroTrans<KernelTransHalf<double>, 1, 4>;
extern template class MacroTrans<KernelTransHalf<double>, 1, 3>;
extern template class MacroTrans<KernelTransHalf<double>, 1, 2>;
extern template class MacroTrans<KernelTransHalf<double>, 1, 1>;

extern template class MacroTrans<KernelTransHalf<FloatComplex>, 4, 1>;
extern template class MacroTrans<KernelTransHalf<FloatComplex>, 3, 1>;
extern template class MacroTrans<KernelTransHalf<FloatComplex>, 2, 1>;
extern template class MacroTrans<KernelTransHalf<FloatComplex>, 1, 4>;
extern template class MacroTrans<KernelTransHalf<FloatComplex>, 1, 3>;
extern template class MacroTrans<KernelTransHalf<FloatComplex>, 1, 2>;
extern template class MacroTrans<KernelTransHalf<FloatComplex>, 1, 1>;

extern template class MacroTrans<KernelTransHalf<DoubleComplex>, 4, 1>;
extern template class MacroTrans<KernelTransHalf<DoubleComplex>, 3, 1>;
extern template class MacroTrans<KernelTransHalf<DoubleComplex>, 2, 1>;
extern template class MacroTrans<KernelTransHalf<DoubleComplex>, 1, 4>;
extern template class MacroTrans<KernelTransHalf<DoubleComplex>, 1, 3>;
extern template class MacroTrans<KernelTransHalf<DoubleComplex>, 1, 2>;
extern template class MacroTrans<KernelTransHalf<DoubleComplex>, 1, 1>;

extern template class MacroTransLinear<float>;
extern template class MacroTransLinear<double>;
extern template class MacroTransLinear<FloatComplex>;
extern template class MacroTransLinear<DoubleComplex>;


/*
 * Explicit template instantiation declaration for class MacroTransScalar
 */
extern template class MacroTransScalar<float>;
extern template class MacroTransScalar<double>;
extern template class MacroTransScalar<FloatComplex>;
extern template class MacroTransScalar<DoubleComplex>;

#endif // HPTT_KERNELS_MACRO_KERNEL_TRANS_TCC_
