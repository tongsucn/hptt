#include <hptc/kernels/macro_kernel_trans.h>

#include <hptc/types.h>
#include <hptc/util/util_trans.h>


namespace hptc {

/*
 * Specialization and implementation for class MacroTrans
 */
template <typename MicroKernel>
MacroTrans<MicroKernel, 4, 4>::MacroTrans()
    : MacroTransData<MicroKernel, 4, 4>() {
}

template <typename MicroKernel>
MacroTrans<MicroKernel, 4, 3>::MacroTrans()
    : MacroTransData<MicroKernel, 4, 3>() {
}

template <typename MicroKernel>
MacroTrans<MicroKernel, 4, 2>::MacroTrans()
    : MacroTransData<MicroKernel, 4, 2>() {
}

template <typename MicroKernel>
MacroTrans<MicroKernel, 4, 1>::MacroTrans()
    : MacroTransData<MicroKernel, 4, 1>() {
}

template <typename MicroKernel>
MacroTrans<MicroKernel, 3, 4>::MacroTrans()
    : MacroTransData<MicroKernel, 3, 4>() {
}

template <typename MicroKernel>
MacroTrans<MicroKernel, 3, 3>::MacroTrans()
    : MacroTransData<MicroKernel, 3, 3>() {
}

template <typename MicroKernel>
MacroTrans<MicroKernel, 3, 2>::MacroTrans()
    : MacroTransData<MicroKernel, 3, 2>() {
}

template <typename MicroKernel>
MacroTrans<MicroKernel, 3, 1>::MacroTrans()
    : MacroTransData<MicroKernel, 3, 1>() {
}

template <typename MicroKernel>
MacroTrans<MicroKernel, 2, 4>::MacroTrans()
    : MacroTransData<MicroKernel, 2, 4>() {
}

template <typename MicroKernel>
MacroTrans<MicroKernel, 2, 3>::MacroTrans()
    : MacroTransData<MicroKernel, 2, 3>() {
}

template <typename MicroKernel>
MacroTrans<MicroKernel, 2, 2>::MacroTrans()
    : MacroTransData<MicroKernel, 2, 2>() {
}

template <typename MicroKernel>
MacroTrans<MicroKernel, 2, 1>::MacroTrans()
    : MacroTransData<MicroKernel, 2, 1>() {
}

template <typename MicroKernel>
MacroTrans<MicroKernel, 1, 4>::MacroTrans()
    : MacroTransData<MicroKernel, 1, 4>() {
}

template <typename MicroKernel>
MacroTrans<MicroKernel, 1, 3>::MacroTrans()
    : MacroTransData<MicroKernel, 1, 3>() {
}

template <typename MicroKernel>
MacroTrans<MicroKernel, 1, 2>::MacroTrans()
    : MacroTransData<MicroKernel, 1, 2>() {
}

template <typename MicroKernel>
MacroTrans<MicroKernel, 1, 1>::MacroTrans()
    : MacroTransData<MicroKernel, 1, 1>() {
}


template <typename MicroKernel>
void MacroTrans<MicroKernel, 4, 4>::exec(
    const typename MicroKernel::Float *data_in,
    typename MicroKernel::Float *data_out, const TensorIdx stride_in_outld,
    const TensorIdx stride_out_inld) const {
  this->kernel_.exec(data_in, data_out, stride_in_outld, stride_out_inld);
  this->kernel_.exec(data_in + this->kn_width_ * stride_in_outld,
      data_out + this->kn_width_, stride_in_outld, stride_out_inld);
  this->kernel_.exec(data_in + 2 * this->kn_width_ * stride_in_outld,
      data_out + 2 * this->kn_width_, stride_in_outld, stride_out_inld);
  this->kernel_.exec(data_in + 3 * this->kn_width_ * stride_in_outld,
      data_out + 3 * this->kn_width_, stride_in_outld, stride_out_inld);
  this->kernel_.exec(data_in + this->kn_width_,
      data_out + this->kn_width_ * stride_out_inld, stride_in_outld,
      stride_out_inld);
  this->kernel_.exec(
      data_in + this->kn_width_ + this->kn_width_ * stride_in_outld,
      data_out + this->kn_width_ * stride_out_inld + this->kn_width_,
      stride_in_outld, stride_out_inld);
  this->kernel_.exec(
      data_in + this->kn_width_ + 2 * this->kn_width_ * stride_in_outld,
      data_out + this->kn_width_ * stride_out_inld + 2 * this->kn_width_,
      stride_in_outld, stride_out_inld);
  this->kernel_.exec(
      data_in + this->kn_width_ + 3 * this->kn_width_ * stride_in_outld,
      data_out + this->kn_width_ * stride_out_inld + 3 * this->kn_width_,
      stride_in_outld, stride_out_inld);
  this->kernel_.exec(data_in + 2 * this->kn_width_,
      data_out + 2 * this->kn_width_ * stride_out_inld, stride_in_outld,
      stride_out_inld);
  this->kernel_.exec(
      data_in + 2 * this->kn_width_ + this->kn_width_ * stride_in_outld,
      data_out + 2 * this->kn_width_ * stride_out_inld + this->kn_width_,
      stride_in_outld, stride_out_inld);
  this->kernel_.exec(
      data_in + 2 * this->kn_width_ + 2 * this->kn_width_ * stride_in_outld,
      data_out + 2 * this->kn_width_ * stride_out_inld + 2 * this->kn_width_,
      stride_in_outld, stride_out_inld);
  this->kernel_.exec(
      data_in + 2 * this->kn_width_ + 3 * this->kn_width_ * stride_in_outld,
      data_out + 2 * this->kn_width_ * stride_out_inld + 3 * this->kn_width_,
      stride_in_outld, stride_out_inld);
  this->kernel_.exec(data_in + 3 * this->kn_width_,
      data_out + 3 * this->kn_width_ * stride_out_inld, stride_in_outld,
      stride_out_inld);
  this->kernel_.exec(
      data_in + 3 * this->kn_width_ + this->kn_width_ * stride_in_outld,
      data_out + 3 * this->kn_width_ * stride_out_inld + this->kn_width_,
      stride_in_outld, stride_out_inld);
  this->kernel_.exec(
      data_in + 3 * this->kn_width_ + 2 * this->kn_width_ * stride_in_outld,
      data_out + 3 * this->kn_width_ * stride_out_inld + 2 * this->kn_width_,
      stride_in_outld, stride_out_inld);
  this->kernel_.exec(
      data_in + 3 * this->kn_width_ + 3 * this->kn_width_ * stride_in_outld,
      data_out + 3 * this->kn_width_ * stride_out_inld + 3 * this->kn_width_,
      stride_in_outld, stride_out_inld);
}

template <typename MicroKernel>
void MacroTrans<MicroKernel, 4, 3>::exec(
    const typename MicroKernel::Float *data_in,
    typename MicroKernel::Float *data_out, const TensorIdx stride_in_outld,
    const TensorIdx stride_out_inld) const {
  this->kernel_.exec(data_in, data_out, stride_in_outld, stride_out_inld);
  this->kernel_.exec(data_in + this->kn_width_ * stride_in_outld,
      data_out + this->kn_width_, stride_in_outld, stride_out_inld);
  this->kernel_.exec(data_in + 2 * this->kn_width_ * stride_in_outld,
      data_out + 2 * this->kn_width_, stride_in_outld, stride_out_inld);
  this->kernel_.exec(data_in + this->kn_width_,
      data_out + this->kn_width_ * stride_out_inld, stride_in_outld,
      stride_out_inld);
  this->kernel_.exec(
      data_in + this->kn_width_ + this->kn_width_ * stride_in_outld,
      data_out + this->kn_width_ * stride_out_inld + this->kn_width_,
      stride_in_outld, stride_out_inld);
  this->kernel_.exec(
      data_in + this->kn_width_ + 2 * this->kn_width_ * stride_in_outld,
      data_out + this->kn_width_ * stride_out_inld + 2 * this->kn_width_,
      stride_in_outld, stride_out_inld);
  this->kernel_.exec(data_in + 2 * this->kn_width_,
      data_out + 2 * this->kn_width_ * stride_out_inld, stride_in_outld,
      stride_out_inld);
  this->kernel_.exec(
      data_in + 2 * this->kn_width_ + this->kn_width_ * stride_in_outld,
      data_out + 2 * this->kn_width_ * stride_out_inld + this->kn_width_,
      stride_in_outld, stride_out_inld);
  this->kernel_.exec(
      data_in + 2 * this->kn_width_ + 2 * this->kn_width_ * stride_in_outld,
      data_out + 2 * this->kn_width_ * stride_out_inld + 2 * this->kn_width_,
      stride_in_outld, stride_out_inld);
  this->kernel_.exec(data_in + 3 * this->kn_width_,
      data_out + 3 * this->kn_width_ * stride_out_inld, stride_in_outld,
      stride_out_inld);
  this->kernel_.exec(
      data_in + 3 * this->kn_width_ + this->kn_width_ * stride_in_outld,
      data_out + 3 * this->kn_width_ * stride_out_inld + this->kn_width_,
      stride_in_outld, stride_out_inld);
  this->kernel_.exec(
      data_in + 3 * this->kn_width_ + 2 * this->kn_width_ * stride_in_outld,
      data_out + 3 * this->kn_width_ * stride_out_inld + 2 * this->kn_width_,
      stride_in_outld, stride_out_inld);
}

template <typename MicroKernel>
void MacroTrans<MicroKernel, 4, 2>::exec(
    const typename MicroKernel::Float *data_in,
    typename MicroKernel::Float *data_out, const TensorIdx stride_in_outld,
    const TensorIdx stride_out_inld) const {
  this->kernel_.exec(data_in, data_out, stride_in_outld, stride_out_inld);
  this->kernel_.exec(data_in + this->kn_width_ * stride_in_outld,
      data_out + this->kn_width_, stride_in_outld, stride_out_inld);
  this->kernel_.exec(data_in + this->kn_width_,
      data_out + this->kn_width_ * stride_out_inld, stride_in_outld,
      stride_out_inld);
  this->kernel_.exec(
      data_in + this->kn_width_ + this->kn_width_ * stride_in_outld,
      data_out + this->kn_width_ * stride_out_inld + this->kn_width_,
      stride_in_outld, stride_out_inld);
  this->kernel_.exec(data_in + 2 * this->kn_width_,
      data_out + 2 * this->kn_width_ * stride_out_inld,
      stride_in_outld, stride_out_inld);
  this->kernel_.exec(
      data_in + 2 * this->kn_width_ + this->kn_width_ * stride_in_outld,
      data_out + 2 * this->kn_width_ * stride_out_inld + this->kn_width_,
      stride_in_outld, stride_out_inld);
  this->kernel_.exec(data_in + 3 * this->kn_width_,
      data_out + 3 * this->kn_width_ * stride_out_inld,
      stride_in_outld, stride_out_inld);
  this->kernel_.exec(
      data_in + 3 * this->kn_width_ + this->kn_width_ * stride_in_outld,
      data_out + 3 * this->kn_width_ * stride_out_inld + this->kn_width_,
      stride_in_outld, stride_out_inld);
}

template <typename MicroKernel>
void MacroTrans<MicroKernel, 4, 1>::exec(
    const typename MicroKernel::Float *data_in,
    typename MicroKernel::Float *data_out, const TensorIdx stride_in_outld,
    const TensorIdx stride_out_inld) const {
  this->kernel_.exec(data_in, data_out, stride_in_outld, stride_out_inld);
  this->kernel_.exec( data_in + this->kn_width_,
      data_out + this->kn_width_ * stride_out_inld, stride_in_outld,
      stride_out_inld);
  this->kernel_.exec(data_in + 2 * this->kn_width_,
      data_out + 2 * this->kn_width_ * stride_out_inld, stride_in_outld,
      stride_out_inld);
  this->kernel_.exec(data_in + 3 * this->kn_width_,
      data_out + 3 * this->kn_width_ * stride_out_inld, stride_in_outld,
      stride_out_inld);
}

template <typename MicroKernel>
void MacroTrans<MicroKernel, 3, 4>::exec(
    const typename MicroKernel::Float *data_in,
    typename MicroKernel::Float *data_out, const TensorIdx stride_in_outld,
    const TensorIdx stride_out_inld) const {
  this->kernel_.exec(data_in, data_out, stride_in_outld, stride_out_inld);
  this->kernel_.exec(data_in + this->kn_width_ * stride_in_outld,
      data_out + this->kn_width_, stride_in_outld, stride_out_inld);
  this->kernel_.exec(data_in + 2 * this->kn_width_ * stride_in_outld,
      data_out + 2 * this->kn_width_, stride_in_outld, stride_out_inld);
  this->kernel_.exec(data_in + 3 * this->kn_width_ * stride_in_outld,
      data_out + 3 * this->kn_width_, stride_in_outld, stride_out_inld);
  this->kernel_.exec( data_in + this->kn_width_,
      data_out + this->kn_width_ * stride_out_inld, stride_in_outld,
      stride_out_inld);
  this->kernel_.exec(
      data_in + this->kn_width_ + this->kn_width_ * stride_in_outld,
      data_out + this->kn_width_ * stride_out_inld + this->kn_width_,
      stride_in_outld, stride_out_inld);
  this->kernel_.exec(
      data_in + this->kn_width_ + 2 * this->kn_width_ * stride_in_outld,
      data_out + this->kn_width_ * stride_out_inld + 2 * this->kn_width_,
      stride_in_outld, stride_out_inld);
  this->kernel_.exec(
      data_in + this->kn_width_ + 3 * this->kn_width_ * stride_in_outld,
      data_out + this->kn_width_ * stride_out_inld + 3 * this->kn_width_,
      stride_in_outld, stride_out_inld);
  this->kernel_.exec(data_in + 2 * this->kn_width_,
      data_out + 2 * this->kn_width_ * stride_out_inld, stride_in_outld,
      stride_out_inld);
  this->kernel_.exec(
      data_in + 2 * this->kn_width_ + this->kn_width_ * stride_in_outld,
      data_out + 2 * this->kn_width_ * stride_out_inld + this->kn_width_,
      stride_in_outld, stride_out_inld);
  this->kernel_.exec(
      data_in + 2 * this->kn_width_ + 2 * this->kn_width_ * stride_in_outld,
      data_out + 2 * this->kn_width_ * stride_out_inld + 2 * this->kn_width_,
      stride_in_outld, stride_out_inld);
  this->kernel_.exec(
      data_in + 2 * this->kn_width_ + 3 * this->kn_width_ * stride_in_outld,
      data_out + 2 * this->kn_width_ * stride_out_inld + 3 * this->kn_width_,
      stride_in_outld, stride_out_inld);
}

template <typename MicroKernel>
void MacroTrans<MicroKernel, 3, 3>::exec(
    const typename MicroKernel::Float *data_in,
    typename MicroKernel::Float *data_out, const TensorIdx stride_in_outld,
    const TensorIdx stride_out_inld) const {
  this->kernel_.exec(data_in, data_out, stride_in_outld, stride_out_inld);
  this->kernel_.exec(data_in + this->kn_width_ * stride_in_outld,
      data_out + this->kn_width_, stride_in_outld, stride_out_inld);
  this->kernel_.exec(data_in + 2 * this->kn_width_ * stride_in_outld,
      data_out + 2 * this->kn_width_, stride_in_outld, stride_out_inld);
  this->kernel_.exec( data_in + this->kn_width_,
      data_out + this->kn_width_ * stride_out_inld, stride_in_outld,
      stride_out_inld);
  this->kernel_.exec(
      data_in + this->kn_width_ + this->kn_width_ * stride_in_outld,
      data_out + this->kn_width_ * stride_out_inld + this->kn_width_,
      stride_in_outld, stride_out_inld);
  this->kernel_.exec(
      data_in + this->kn_width_ + 2 * this->kn_width_ * stride_in_outld,
      data_out + this->kn_width_ * stride_out_inld + 2 * this->kn_width_,
      stride_in_outld, stride_out_inld);
  this->kernel_.exec(data_in + 2 * this->kn_width_,
      data_out + 2 * this->kn_width_ * stride_out_inld, stride_in_outld,
      stride_out_inld);
  this->kernel_.exec(
      data_in + 2 * this->kn_width_ + this->kn_width_ * stride_in_outld,
      data_out + 2 * this->kn_width_ * stride_out_inld + this->kn_width_,
      stride_in_outld, stride_out_inld);
  this->kernel_.exec(
      data_in + 2 * this->kn_width_ + 2 * this->kn_width_ * stride_in_outld,
      data_out + 2 * this->kn_width_ * stride_out_inld + 2 * this->kn_width_,
      stride_in_outld, stride_out_inld);
}

template <typename MicroKernel>
void MacroTrans<MicroKernel, 3, 2>::exec(
    const typename MicroKernel::Float *data_in,
    typename MicroKernel::Float *data_out, const TensorIdx stride_in_outld,
    const TensorIdx stride_out_inld) const {
  this->kernel_.exec(data_in, data_out, stride_in_outld, stride_out_inld);
  this->kernel_.exec(data_in + this->kn_width_ * stride_in_outld,
      data_out + this->kn_width_, stride_in_outld, stride_out_inld);
  this->kernel_.exec(data_in + this->kn_width_,
      data_out + this->kn_width_ * stride_out_inld, stride_in_outld,
      stride_out_inld);
  this->kernel_.exec(
      data_in + this->kn_width_ + this->kn_width_ * stride_in_outld,
      data_out + this->kn_width_ * stride_out_inld + this->kn_width_,
      stride_in_outld, stride_out_inld);
  this->kernel_.exec(data_in + 2 * this->kn_width_,
      data_out + 2 * this->kn_width_ * stride_out_inld,
      stride_in_outld, stride_out_inld);
  this->kernel_.exec(
      data_in + 2 * this->kn_width_ + this->kn_width_ * stride_in_outld,
      data_out + 2 * this->kn_width_ * stride_out_inld + this->kn_width_,
      stride_in_outld, stride_out_inld);
}

template <typename MicroKernel>
void MacroTrans<MicroKernel, 3, 1>::exec(
    const typename MicroKernel::Float *data_in,
    typename MicroKernel::Float *data_out, const TensorIdx stride_in_outld,
    const TensorIdx stride_out_inld) const {
  this->kernel_.exec(data_in, data_out, stride_in_outld, stride_out_inld);
  this->kernel_.exec(data_in + this->kn_width_,
      data_out + this->kn_width_ * stride_out_inld, stride_in_outld,
      stride_out_inld);
  this->kernel_.exec(data_in + 2 * this->kn_width_,
      data_out + 2 * this->kn_width_ * stride_out_inld, stride_in_outld,
      stride_out_inld);
}

template <typename MicroKernel>
void MacroTrans<MicroKernel, 2, 4>::exec(
    const typename MicroKernel::Float *data_in,
    typename MicroKernel::Float *data_out, const TensorIdx stride_in_outld,
    const TensorIdx stride_out_inld) const {
  this->kernel_.exec(data_in, data_out, stride_in_outld, stride_out_inld);
  this->kernel_.exec(data_in + this->kn_width_ * stride_in_outld,
      data_out + this->kn_width_, stride_in_outld, stride_out_inld);
  this->kernel_.exec(data_in + 2 * this->kn_width_ * stride_in_outld,
      data_out + 2 * this->kn_width_, stride_in_outld, stride_out_inld);
  this->kernel_.exec(data_in + 3 * this->kn_width_ * stride_in_outld,
      data_out + 3 * this->kn_width_, stride_in_outld, stride_out_inld);
  this->kernel_.exec(data_in + this->kn_width_,
      data_out + this->kn_width_ * stride_out_inld, stride_in_outld,
      stride_out_inld);
  this->kernel_.exec(
      data_in + this->kn_width_ + this->kn_width_ * stride_in_outld,
      data_out + this->kn_width_ * stride_out_inld + this->kn_width_,
      stride_in_outld, stride_out_inld);
  this->kernel_.exec(
      data_in + this->kn_width_ + 2 * this->kn_width_ * stride_in_outld,
      data_out + this->kn_width_ * stride_out_inld + 2 * this->kn_width_,
      stride_in_outld, stride_out_inld);
  this->kernel_.exec(
      data_in + this->kn_width_ + 3 * this->kn_width_ * stride_in_outld,
      data_out + this->kn_width_ * stride_out_inld + 3 * this->kn_width_,
      stride_in_outld, stride_out_inld);
}

template <typename MicroKernel>
void MacroTrans<MicroKernel, 2, 3>::exec(
    const typename MicroKernel::Float *data_in,
    typename MicroKernel::Float *data_out, const TensorIdx stride_in_outld,
    const TensorIdx stride_out_inld) const {
  this->kernel_.exec(data_in, data_out, stride_in_outld, stride_out_inld);
  this->kernel_.exec(data_in + this->kn_width_ * stride_in_outld,
      data_out + this->kn_width_, stride_in_outld, stride_out_inld);
  this->kernel_.exec(data_in + 2 * this->kn_width_ * stride_in_outld,
      data_out + 2 * this->kn_width_, stride_in_outld, stride_out_inld);
  this->kernel_.exec(data_in + this->kn_width_,
      data_out + this->kn_width_ * stride_out_inld, stride_in_outld,
      stride_out_inld);
  this->kernel_.exec(
      data_in + this->kn_width_ + this->kn_width_ * stride_in_outld,
      data_out + this->kn_width_ * stride_out_inld + this->kn_width_,
      stride_in_outld, stride_out_inld);
  this->kernel_.exec(
      data_in + this->kn_width_ + 2 * this->kn_width_ * stride_in_outld,
      data_out + this->kn_width_ * stride_out_inld + 2 * this->kn_width_,
      stride_in_outld, stride_out_inld);
}

template <typename MicroKernel>
void MacroTrans<MicroKernel, 2, 2>::exec(
    const typename MicroKernel::Float *data_in,
    typename MicroKernel::Float *data_out, const TensorIdx stride_in_outld,
    const TensorIdx stride_out_inld) const {
  this->kernel_.exec(data_in, data_out, stride_in_outld, stride_out_inld);
  this->kernel_.exec(data_in + this->kn_width_ * stride_in_outld,
      data_out + this->kn_width_, stride_in_outld, stride_out_inld);
  this->kernel_.exec(data_in + this->kn_width_,
      data_out + this->kn_width_ * stride_out_inld, stride_in_outld,
      stride_out_inld);
  this->kernel_.exec(
      data_in + this->kn_width_ + this->kn_width_ * stride_in_outld,
      data_out + this->kn_width_ * stride_out_inld + this->kn_width_,
      stride_in_outld, stride_out_inld);
}

template <typename MicroKernel>
void MacroTrans<MicroKernel, 2, 1>::exec(
    const typename MicroKernel::Float *data_in,
    typename MicroKernel::Float *data_out, const TensorIdx stride_in_outld,
    const TensorIdx stride_out_inld) const {
  this->kernel_.exec(data_in, data_out, stride_in_outld, stride_out_inld);
  this->kernel_.exec(data_in + this->kn_width_,
      data_out + this->kn_width_ * stride_out_inld, stride_in_outld,
      stride_out_inld);
}

template <typename MicroKernel>
void MacroTrans<MicroKernel, 1, 4>::exec(
    const typename MicroKernel::Float *data_in,
    typename MicroKernel::Float *data_out, const TensorIdx stride_in_outld,
    const TensorIdx stride_out_inld) const {
  this->kernel_.exec(data_in, data_out, stride_in_outld, stride_out_inld);
  this->kernel_.exec(data_in + this->kn_width_ * stride_in_outld,
      data_out + this->kn_width_, stride_in_outld, stride_out_inld);
  this->kernel_.exec(data_in + 2 * this->kn_width_ * stride_in_outld,
      data_out + 2 * this->kn_width_, stride_in_outld, stride_out_inld);
  this->kernel_.exec(data_in + 3 * this->kn_width_ * stride_in_outld,
      data_out + 3 * this->kn_width_, stride_in_outld, stride_out_inld);
}

template <typename MicroKernel>
void MacroTrans<MicroKernel, 1, 3>::exec(
    const typename MicroKernel::Float *data_in,
    typename MicroKernel::Float *data_out, const TensorIdx stride_in_outld,
    const TensorIdx stride_out_inld) const {
  this->kernel_.exec(data_in, data_out, stride_in_outld, stride_out_inld);
  this->kernel_.exec(data_in + this->kn_width_ * stride_in_outld,
      data_out + this->kn_width_, stride_in_outld, stride_out_inld);
  this->kernel_.exec(data_in + 2 * this->kn_width_ * stride_in_outld,
      data_out + 2 * this->kn_width_, stride_in_outld, stride_out_inld);
}

template <typename MicroKernel>
void MacroTrans<MicroKernel, 1, 2>::exec(
    const typename MicroKernel::Float *data_in,
    typename MicroKernel::Float *data_out, const TensorIdx stride_in_outld,
    const TensorIdx stride_out_inld) const {
  this->kernel_.exec(data_in, data_out, stride_in_outld, stride_out_inld);
  this->kernel_.exec(data_in + this->kn_width_ * stride_in_outld,
      data_out + this->kn_width_, stride_in_outld, stride_out_inld);
}

template <typename MicroKernel>
void MacroTrans<MicroKernel, 1, 1>::exec(
    const typename MicroKernel::Float *data_in,
    typename MicroKernel::Float *data_out, const TensorIdx stride_in_outld,
    const TensorIdx stride_out_inld) const {
  this->kernel_.exec(data_in, data_out, stride_in_outld, stride_out_inld);
}


/*
 * Implementation for class MacroTransLinear
 */
template <typename FloatType>
void MacroTransLinear<FloatType>::set_coef(
    const DeducedFloatType<FloatType> alpha,
    const DeducedFloatType<FloatType> beta) {
  this->kernel_.set_coef(alpha, beta);
}


template <typename FloatType>
void MacroTransLinear<FloatType>::set_wrapper_loop(
    const TensorIdx stride_in_inld,
    const TensorIdx stride_in_outld, const TensorIdx stride_out_inld,
    const TensorIdx stride_out_outld, const TensorUInt size_kn_inld,
    const TensorUInt size_kn_outld) {
  this->kernel_.set_wrapper_loop(stride_in_inld, stride_in_outld,
      stride_out_inld, stride_out_outld, size_kn_inld, size_kn_outld);
}


template <typename FloatType>
void MacroTransLinear<FloatType>::exec(const FloatType *data_in,
    FloatType *data_out, const TensorIdx size_trans,
    const TensorIdx size_pad) const {
  this->kernel_.exec(data_in, data_out, size_trans, size_pad);
}


/*
 * Implementation for class MacroTransScalar
 */
template <typename FloatType>
void MacroTransScalar<FloatType>::set_coef(
    const DeducedFloatType<FloatType> alpha,
    const DeducedFloatType<FloatType> beta) {
  this->alpha_ = alpha, this->beta_ = beta;
}


template <typename FloatType>
void MacroTransScalar<FloatType>::exec(const FloatType *data_in,
    FloatType *data_out, const TensorIdx, const TensorIdx) const {
  *data_out = this->alpha_ * *data_in + this->beta_ * *data_out;
}


/*
 * Explicit template instantiation for class MacroTrans
 */
template class MacroTrans<KernelTransFull<float>, 4, 4>;
template class MacroTrans<KernelTransFull<float>, 4, 3>;
template class MacroTrans<KernelTransFull<float>, 4, 2>;
template class MacroTrans<KernelTransFull<float>, 4, 1>;
template class MacroTrans<KernelTransFull<float>, 3, 4>;
template class MacroTrans<KernelTransFull<float>, 3, 3>;
template class MacroTrans<KernelTransFull<float>, 3, 2>;
template class MacroTrans<KernelTransFull<float>, 3, 1>;
template class MacroTrans<KernelTransFull<float>, 2, 4>;
template class MacroTrans<KernelTransFull<float>, 2, 3>;
template class MacroTrans<KernelTransFull<float>, 2, 2>;
template class MacroTrans<KernelTransFull<float>, 2, 1>;
template class MacroTrans<KernelTransFull<float>, 1, 4>;
template class MacroTrans<KernelTransFull<float>, 1, 3>;
template class MacroTrans<KernelTransFull<float>, 1, 2>;
template class MacroTrans<KernelTransFull<float>, 1, 1>;

template class MacroTrans<KernelTransFull<double>, 4, 4>;
template class MacroTrans<KernelTransFull<double>, 4, 3>;
template class MacroTrans<KernelTransFull<double>, 4, 2>;
template class MacroTrans<KernelTransFull<double>, 4, 1>;
template class MacroTrans<KernelTransFull<double>, 3, 4>;
template class MacroTrans<KernelTransFull<double>, 3, 3>;
template class MacroTrans<KernelTransFull<double>, 3, 2>;
template class MacroTrans<KernelTransFull<double>, 3, 1>;
template class MacroTrans<KernelTransFull<double>, 2, 4>;
template class MacroTrans<KernelTransFull<double>, 2, 3>;
template class MacroTrans<KernelTransFull<double>, 2, 2>;
template class MacroTrans<KernelTransFull<double>, 2, 1>;
template class MacroTrans<KernelTransFull<double>, 1, 4>;
template class MacroTrans<KernelTransFull<double>, 1, 3>;
template class MacroTrans<KernelTransFull<double>, 1, 2>;
template class MacroTrans<KernelTransFull<double>, 1, 1>;

template class MacroTrans<KernelTransFull<FloatComplex>, 4, 4>;
template class MacroTrans<KernelTransFull<FloatComplex>, 4, 3>;
template class MacroTrans<KernelTransFull<FloatComplex>, 4, 2>;
template class MacroTrans<KernelTransFull<FloatComplex>, 4, 1>;
template class MacroTrans<KernelTransFull<FloatComplex>, 3, 4>;
template class MacroTrans<KernelTransFull<FloatComplex>, 3, 3>;
template class MacroTrans<KernelTransFull<FloatComplex>, 3, 2>;
template class MacroTrans<KernelTransFull<FloatComplex>, 3, 1>;
template class MacroTrans<KernelTransFull<FloatComplex>, 2, 4>;
template class MacroTrans<KernelTransFull<FloatComplex>, 2, 3>;
template class MacroTrans<KernelTransFull<FloatComplex>, 2, 2>;
template class MacroTrans<KernelTransFull<FloatComplex>, 2, 1>;
template class MacroTrans<KernelTransFull<FloatComplex>, 1, 4>;
template class MacroTrans<KernelTransFull<FloatComplex>, 1, 3>;
template class MacroTrans<KernelTransFull<FloatComplex>, 1, 2>;
template class MacroTrans<KernelTransFull<FloatComplex>, 1, 1>;

template class MacroTrans<KernelTransFull<DoubleComplex>, 4, 4>;
template class MacroTrans<KernelTransFull<DoubleComplex>, 4, 3>;
template class MacroTrans<KernelTransFull<DoubleComplex>, 4, 2>;
template class MacroTrans<KernelTransFull<DoubleComplex>, 4, 1>;
template class MacroTrans<KernelTransFull<DoubleComplex>, 3, 4>;
template class MacroTrans<KernelTransFull<DoubleComplex>, 3, 3>;
template class MacroTrans<KernelTransFull<DoubleComplex>, 3, 2>;
template class MacroTrans<KernelTransFull<DoubleComplex>, 3, 1>;
template class MacroTrans<KernelTransFull<DoubleComplex>, 2, 4>;
template class MacroTrans<KernelTransFull<DoubleComplex>, 2, 3>;
template class MacroTrans<KernelTransFull<DoubleComplex>, 2, 2>;
template class MacroTrans<KernelTransFull<DoubleComplex>, 2, 1>;
template class MacroTrans<KernelTransFull<DoubleComplex>, 1, 4>;
template class MacroTrans<KernelTransFull<DoubleComplex>, 1, 3>;
template class MacroTrans<KernelTransFull<DoubleComplex>, 1, 2>;
template class MacroTrans<KernelTransFull<DoubleComplex>, 1, 1>;

template class MacroTrans<KernelTransHalf<float>, 4, 1>;
template class MacroTrans<KernelTransHalf<float>, 3, 1>;
template class MacroTrans<KernelTransHalf<float>, 2, 1>;
template class MacroTrans<KernelTransHalf<float>, 1, 4>;
template class MacroTrans<KernelTransHalf<float>, 1, 3>;
template class MacroTrans<KernelTransHalf<float>, 1, 2>;
template class MacroTrans<KernelTransHalf<float>, 1, 1>;

template class MacroTrans<KernelTransHalf<double>, 4, 1>;
template class MacroTrans<KernelTransHalf<double>, 3, 1>;
template class MacroTrans<KernelTransHalf<double>, 2, 1>;
template class MacroTrans<KernelTransHalf<double>, 1, 4>;
template class MacroTrans<KernelTransHalf<double>, 1, 3>;
template class MacroTrans<KernelTransHalf<double>, 1, 2>;
template class MacroTrans<KernelTransHalf<double>, 1, 1>;

template class MacroTrans<KernelTransHalf<FloatComplex>, 4, 1>;
template class MacroTrans<KernelTransHalf<FloatComplex>, 3, 1>;
template class MacroTrans<KernelTransHalf<FloatComplex>, 2, 1>;
template class MacroTrans<KernelTransHalf<FloatComplex>, 1, 4>;
template class MacroTrans<KernelTransHalf<FloatComplex>, 1, 3>;
template class MacroTrans<KernelTransHalf<FloatComplex>, 1, 2>;
template class MacroTrans<KernelTransHalf<FloatComplex>, 1, 1>;

template class MacroTrans<KernelTransHalf<DoubleComplex>, 4, 1>;
template class MacroTrans<KernelTransHalf<DoubleComplex>, 3, 1>;
template class MacroTrans<KernelTransHalf<DoubleComplex>, 2, 1>;
template class MacroTrans<KernelTransHalf<DoubleComplex>, 1, 4>;
template class MacroTrans<KernelTransHalf<DoubleComplex>, 1, 3>;
template class MacroTrans<KernelTransHalf<DoubleComplex>, 1, 2>;
template class MacroTrans<KernelTransHalf<DoubleComplex>, 1, 1>;

template class MacroTransLinear<float>;
template class MacroTransLinear<double>;
template class MacroTransLinear<FloatComplex>;
template class MacroTransLinear<DoubleComplex>;

/*
 * Explicit template instantiation for class MacroTransScalar
 */
template class MacroTransScalar<float>;
template class MacroTransScalar<double>;
template class MacroTransScalar<FloatComplex>;
template class MacroTransScalar<DoubleComplex>;

}
