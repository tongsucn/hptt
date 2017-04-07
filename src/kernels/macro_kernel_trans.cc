#include <hptt/kernels/macro_kernel_trans.h>

#include <hptt/types.h>
#include <hptt/util/util_trans.h>


namespace hptt {

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
template <typename FloatType,
          bool UPDATE_OUT>
void MacroTransLinear<FloatType, UPDATE_OUT>::set_coef(
    const DeducedFloatType<FloatType> alpha,
    const DeducedFloatType<FloatType> beta) {
  this->kernel_.set_coef(alpha, beta);
}


template <typename FloatType,
          bool UPDATE_OUT>
void MacroTransLinear<FloatType, UPDATE_OUT>::set_wrapper_loop(
    const TensorIdx stride_in_inld,
    const TensorIdx stride_in_outld, const TensorIdx stride_out_inld,
    const TensorIdx stride_out_outld, const TensorUInt size_kn_inld,
    const TensorUInt size_kn_outld) {
  this->kernel_.set_wrapper_loop(stride_in_inld, stride_in_outld,
      stride_out_inld, stride_out_outld, size_kn_inld, size_kn_outld);
}


template <typename FloatType,
          bool UPDATE_OUT>
void MacroTransLinear<FloatType, UPDATE_OUT>::exec(const FloatType *data_in,
    FloatType *data_out, const TensorIdx size_trans,
    const TensorIdx size_pad) const {
  this->kernel_.exec(data_in, data_out, size_trans, size_pad);
}


/*
 * Implementation for class MacroTransScalar
 */
template <typename FloatType,
          bool UPDATE_OUT>
void MacroTransScalar<FloatType, UPDATE_OUT>::set_coef(
    const DeducedFloatType<FloatType> alpha,
    const DeducedFloatType<FloatType> beta) {
  this->alpha_ = alpha, this->beta_ = beta;
}


template <typename FloatType,
          bool UPDATE_OUT>
void MacroTransScalar<FloatType, UPDATE_OUT>::exec(const FloatType *data_in,
    FloatType *data_out, const TensorIdx, const TensorIdx) const {
  *data_out = this->alpha_ * *data_in + this->beta_ * *data_out;
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
