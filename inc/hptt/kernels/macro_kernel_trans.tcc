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
extern template class MacroTrans<KernelTransFull<float, true>, 4, 4>;
extern template class MacroTrans<KernelTransFull<float, true>, 4, 3>;
extern template class MacroTrans<KernelTransFull<float, true>, 4, 2>;
extern template class MacroTrans<KernelTransFull<float, true>, 4, 1>;
extern template class MacroTrans<KernelTransFull<float, true>, 3, 4>;
extern template class MacroTrans<KernelTransFull<float, true>, 3, 3>;
extern template class MacroTrans<KernelTransFull<float, true>, 3, 2>;
extern template class MacroTrans<KernelTransFull<float, true>, 3, 1>;
extern template class MacroTrans<KernelTransFull<float, true>, 2, 4>;
extern template class MacroTrans<KernelTransFull<float, true>, 2, 3>;
extern template class MacroTrans<KernelTransFull<float, true>, 2, 2>;
extern template class MacroTrans<KernelTransFull<float, true>, 2, 1>;
extern template class MacroTrans<KernelTransFull<float, true>, 1, 4>;
extern template class MacroTrans<KernelTransFull<float, true>, 1, 3>;
extern template class MacroTrans<KernelTransFull<float, true>, 1, 2>;
extern template class MacroTrans<KernelTransFull<float, true>, 1, 1>;

extern template class MacroTrans<KernelTransFull<double, true>, 4, 4>;
extern template class MacroTrans<KernelTransFull<double, true>, 4, 3>;
extern template class MacroTrans<KernelTransFull<double, true>, 4, 2>;
extern template class MacroTrans<KernelTransFull<double, true>, 4, 1>;
extern template class MacroTrans<KernelTransFull<double, true>, 3, 4>;
extern template class MacroTrans<KernelTransFull<double, true>, 3, 3>;
extern template class MacroTrans<KernelTransFull<double, true>, 3, 2>;
extern template class MacroTrans<KernelTransFull<double, true>, 3, 1>;
extern template class MacroTrans<KernelTransFull<double, true>, 2, 4>;
extern template class MacroTrans<KernelTransFull<double, true>, 2, 3>;
extern template class MacroTrans<KernelTransFull<double, true>, 2, 2>;
extern template class MacroTrans<KernelTransFull<double, true>, 2, 1>;
extern template class MacroTrans<KernelTransFull<double, true>, 1, 4>;
extern template class MacroTrans<KernelTransFull<double, true>, 1, 3>;
extern template class MacroTrans<KernelTransFull<double, true>, 1, 2>;
extern template class MacroTrans<KernelTransFull<double, true>, 1, 1>;

extern template class MacroTrans<KernelTransFull<FloatComplex, true>, 4, 4>;
extern template class MacroTrans<KernelTransFull<FloatComplex, true>, 4, 3>;
extern template class MacroTrans<KernelTransFull<FloatComplex, true>, 4, 2>;
extern template class MacroTrans<KernelTransFull<FloatComplex, true>, 4, 1>;
extern template class MacroTrans<KernelTransFull<FloatComplex, true>, 3, 4>;
extern template class MacroTrans<KernelTransFull<FloatComplex, true>, 3, 3>;
extern template class MacroTrans<KernelTransFull<FloatComplex, true>, 3, 2>;
extern template class MacroTrans<KernelTransFull<FloatComplex, true>, 3, 1>;
extern template class MacroTrans<KernelTransFull<FloatComplex, true>, 2, 4>;
extern template class MacroTrans<KernelTransFull<FloatComplex, true>, 2, 3>;
extern template class MacroTrans<KernelTransFull<FloatComplex, true>, 2, 2>;
extern template class MacroTrans<KernelTransFull<FloatComplex, true>, 2, 1>;
extern template class MacroTrans<KernelTransFull<FloatComplex, true>, 1, 4>;
extern template class MacroTrans<KernelTransFull<FloatComplex, true>, 1, 3>;
extern template class MacroTrans<KernelTransFull<FloatComplex, true>, 1, 2>;
extern template class MacroTrans<KernelTransFull<FloatComplex, true>, 1, 1>;

extern template class MacroTrans<KernelTransFull<DoubleComplex, true>, 4, 4>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, true>, 4, 3>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, true>, 4, 2>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, true>, 4, 1>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, true>, 3, 4>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, true>, 3, 3>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, true>, 3, 2>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, true>, 3, 1>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, true>, 2, 4>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, true>, 2, 3>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, true>, 2, 2>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, true>, 2, 1>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, true>, 1, 4>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, true>, 1, 3>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, true>, 1, 2>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, true>, 1, 1>;

extern template class MacroTrans<KernelTransHalf<float, true>, 4, 1>;
extern template class MacroTrans<KernelTransHalf<float, true>, 3, 1>;
extern template class MacroTrans<KernelTransHalf<float, true>, 2, 1>;
extern template class MacroTrans<KernelTransHalf<float, true>, 1, 4>;
extern template class MacroTrans<KernelTransHalf<float, true>, 1, 3>;
extern template class MacroTrans<KernelTransHalf<float, true>, 1, 2>;
extern template class MacroTrans<KernelTransHalf<float, true>, 1, 1>;

extern template class MacroTrans<KernelTransHalf<double, true>, 4, 1>;
extern template class MacroTrans<KernelTransHalf<double, true>, 3, 1>;
extern template class MacroTrans<KernelTransHalf<double, true>, 2, 1>;
extern template class MacroTrans<KernelTransHalf<double, true>, 1, 4>;
extern template class MacroTrans<KernelTransHalf<double, true>, 1, 3>;
extern template class MacroTrans<KernelTransHalf<double, true>, 1, 2>;
extern template class MacroTrans<KernelTransHalf<double, true>, 1, 1>;

extern template class MacroTrans<KernelTransHalf<FloatComplex, true>, 4, 1>;
extern template class MacroTrans<KernelTransHalf<FloatComplex, true>, 3, 1>;
extern template class MacroTrans<KernelTransHalf<FloatComplex, true>, 2, 1>;
extern template class MacroTrans<KernelTransHalf<FloatComplex, true>, 1, 4>;
extern template class MacroTrans<KernelTransHalf<FloatComplex, true>, 1, 3>;
extern template class MacroTrans<KernelTransHalf<FloatComplex, true>, 1, 2>;
extern template class MacroTrans<KernelTransHalf<FloatComplex, true>, 1, 1>;

extern template class MacroTrans<KernelTransHalf<DoubleComplex, true>, 4, 1>;
extern template class MacroTrans<KernelTransHalf<DoubleComplex, true>, 3, 1>;
extern template class MacroTrans<KernelTransHalf<DoubleComplex, true>, 2, 1>;
extern template class MacroTrans<KernelTransHalf<DoubleComplex, true>, 1, 4>;
extern template class MacroTrans<KernelTransHalf<DoubleComplex, true>, 1, 3>;
extern template class MacroTrans<KernelTransHalf<DoubleComplex, true>, 1, 2>;
extern template class MacroTrans<KernelTransHalf<DoubleComplex, true>, 1, 1>;

extern template class MacroTrans<KernelTransFull<float, false>, 4, 4>;
extern template class MacroTrans<KernelTransFull<float, false>, 4, 3>;
extern template class MacroTrans<KernelTransFull<float, false>, 4, 2>;
extern template class MacroTrans<KernelTransFull<float, false>, 4, 1>;
extern template class MacroTrans<KernelTransFull<float, false>, 3, 4>;
extern template class MacroTrans<KernelTransFull<float, false>, 3, 3>;
extern template class MacroTrans<KernelTransFull<float, false>, 3, 2>;
extern template class MacroTrans<KernelTransFull<float, false>, 3, 1>;
extern template class MacroTrans<KernelTransFull<float, false>, 2, 4>;
extern template class MacroTrans<KernelTransFull<float, false>, 2, 3>;
extern template class MacroTrans<KernelTransFull<float, false>, 2, 2>;
extern template class MacroTrans<KernelTransFull<float, false>, 2, 1>;
extern template class MacroTrans<KernelTransFull<float, false>, 1, 4>;
extern template class MacroTrans<KernelTransFull<float, false>, 1, 3>;
extern template class MacroTrans<KernelTransFull<float, false>, 1, 2>;
extern template class MacroTrans<KernelTransFull<float, false>, 1, 1>;

extern template class MacroTrans<KernelTransFull<double, false>, 4, 4>;
extern template class MacroTrans<KernelTransFull<double, false>, 4, 3>;
extern template class MacroTrans<KernelTransFull<double, false>, 4, 2>;
extern template class MacroTrans<KernelTransFull<double, false>, 4, 1>;
extern template class MacroTrans<KernelTransFull<double, false>, 3, 4>;
extern template class MacroTrans<KernelTransFull<double, false>, 3, 3>;
extern template class MacroTrans<KernelTransFull<double, false>, 3, 2>;
extern template class MacroTrans<KernelTransFull<double, false>, 3, 1>;
extern template class MacroTrans<KernelTransFull<double, false>, 2, 4>;
extern template class MacroTrans<KernelTransFull<double, false>, 2, 3>;
extern template class MacroTrans<KernelTransFull<double, false>, 2, 2>;
extern template class MacroTrans<KernelTransFull<double, false>, 2, 1>;
extern template class MacroTrans<KernelTransFull<double, false>, 1, 4>;
extern template class MacroTrans<KernelTransFull<double, false>, 1, 3>;
extern template class MacroTrans<KernelTransFull<double, false>, 1, 2>;
extern template class MacroTrans<KernelTransFull<double, false>, 1, 1>;

extern template class MacroTrans<KernelTransFull<FloatComplex, false>, 4, 4>;
extern template class MacroTrans<KernelTransFull<FloatComplex, false>, 4, 3>;
extern template class MacroTrans<KernelTransFull<FloatComplex, false>, 4, 2>;
extern template class MacroTrans<KernelTransFull<FloatComplex, false>, 4, 1>;
extern template class MacroTrans<KernelTransFull<FloatComplex, false>, 3, 4>;
extern template class MacroTrans<KernelTransFull<FloatComplex, false>, 3, 3>;
extern template class MacroTrans<KernelTransFull<FloatComplex, false>, 3, 2>;
extern template class MacroTrans<KernelTransFull<FloatComplex, false>, 3, 1>;
extern template class MacroTrans<KernelTransFull<FloatComplex, false>, 2, 4>;
extern template class MacroTrans<KernelTransFull<FloatComplex, false>, 2, 3>;
extern template class MacroTrans<KernelTransFull<FloatComplex, false>, 2, 2>;
extern template class MacroTrans<KernelTransFull<FloatComplex, false>, 2, 1>;
extern template class MacroTrans<KernelTransFull<FloatComplex, false>, 1, 4>;
extern template class MacroTrans<KernelTransFull<FloatComplex, false>, 1, 3>;
extern template class MacroTrans<KernelTransFull<FloatComplex, false>, 1, 2>;
extern template class MacroTrans<KernelTransFull<FloatComplex, false>, 1, 1>;

extern template class MacroTrans<KernelTransFull<DoubleComplex, false>, 4, 4>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, false>, 4, 3>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, false>, 4, 2>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, false>, 4, 1>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, false>, 3, 4>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, false>, 3, 3>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, false>, 3, 2>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, false>, 3, 1>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, false>, 2, 4>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, false>, 2, 3>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, false>, 2, 2>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, false>, 2, 1>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, false>, 1, 4>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, false>, 1, 3>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, false>, 1, 2>;
extern template class MacroTrans<KernelTransFull<DoubleComplex, false>, 1, 1>;

extern template class MacroTrans<KernelTransHalf<float, false>, 4, 1>;
extern template class MacroTrans<KernelTransHalf<float, false>, 3, 1>;
extern template class MacroTrans<KernelTransHalf<float, false>, 2, 1>;
extern template class MacroTrans<KernelTransHalf<float, false>, 1, 4>;
extern template class MacroTrans<KernelTransHalf<float, false>, 1, 3>;
extern template class MacroTrans<KernelTransHalf<float, false>, 1, 2>;
extern template class MacroTrans<KernelTransHalf<float, false>, 1, 1>;

extern template class MacroTrans<KernelTransHalf<double, false>, 4, 1>;
extern template class MacroTrans<KernelTransHalf<double, false>, 3, 1>;
extern template class MacroTrans<KernelTransHalf<double, false>, 2, 1>;
extern template class MacroTrans<KernelTransHalf<double, false>, 1, 4>;
extern template class MacroTrans<KernelTransHalf<double, false>, 1, 3>;
extern template class MacroTrans<KernelTransHalf<double, false>, 1, 2>;
extern template class MacroTrans<KernelTransHalf<double, false>, 1, 1>;

extern template class MacroTrans<KernelTransHalf<FloatComplex, false>, 4, 1>;
extern template class MacroTrans<KernelTransHalf<FloatComplex, false>, 3, 1>;
extern template class MacroTrans<KernelTransHalf<FloatComplex, false>, 2, 1>;
extern template class MacroTrans<KernelTransHalf<FloatComplex, false>, 1, 4>;
extern template class MacroTrans<KernelTransHalf<FloatComplex, false>, 1, 3>;
extern template class MacroTrans<KernelTransHalf<FloatComplex, false>, 1, 2>;
extern template class MacroTrans<KernelTransHalf<FloatComplex, false>, 1, 1>;

extern template class MacroTrans<KernelTransHalf<DoubleComplex, false>, 4, 1>;
extern template class MacroTrans<KernelTransHalf<DoubleComplex, false>, 3, 1>;
extern template class MacroTrans<KernelTransHalf<DoubleComplex, false>, 2, 1>;
extern template class MacroTrans<KernelTransHalf<DoubleComplex, false>, 1, 4>;
extern template class MacroTrans<KernelTransHalf<DoubleComplex, false>, 1, 3>;
extern template class MacroTrans<KernelTransHalf<DoubleComplex, false>, 1, 2>;
extern template class MacroTrans<KernelTransHalf<DoubleComplex, false>, 1, 1>;


/*
 * Explicit template instantiation for class MacroTransLinear
 */
extern template class MacroTransLinear<float, true>;
extern template class MacroTransLinear<double, true>;
extern template class MacroTransLinear<FloatComplex, true>;
extern template class MacroTransLinear<DoubleComplex, true>;

extern template class MacroTransLinear<float, false>;
extern template class MacroTransLinear<double, false>;
extern template class MacroTransLinear<FloatComplex, false>;
extern template class MacroTransLinear<DoubleComplex, false>;


/*
 * Explicit template instantiation for class MacroTransScalar
 */
extern template class MacroTransScalar<float, true>;
extern template class MacroTransScalar<double, true>;
extern template class MacroTransScalar<FloatComplex, true>;
extern template class MacroTransScalar<DoubleComplex, true>;

extern template class MacroTransScalar<float, false>;
extern template class MacroTransScalar<double, false>;
extern template class MacroTransScalar<FloatComplex, false>;
extern template class MacroTransScalar<DoubleComplex, false>;

#endif // HPTT_KERNELS_MACRO_KERNEL_TRANS_TCC_
