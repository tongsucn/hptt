#include <hptc/kernels/micro_kernel_trans.h>

#include <type_traits>

#include <hptc/arch/arch.h>
#include <hptc/util/util_trans.h>


namespace hptc {

/*
 * Implementation of class KernelTransProxyBase
 */
KernelTransProxyBase::KernelTransProxyBase()
    : lib_loader_(LibLoader::get_loader()) {
}


/*
 * Implementation of class KernelTransProxy
 */
template <typename FloatType,
          KernelTypeTrans TYPE>
KernelTransProxy<FloatType, TYPE>::KernelTransProxy()
    : KernelTransProxyBase(),
      set_reg_impl_(nullptr),
      exec_impl_(nullptr),
      kn_width_(0) {
  // Initialize function pointers
  this->init_func_ptr_<FloatType>(FloatType(0));

  // Initialize kernel width
  if (KernelTypeTrans::KERNEL_FULL == TYPE) {
    this->kn_width_ = *reinterpret_cast<const TensorUInt *>(
        this->lib_loader_.dlsym("REG_SIZE")) / sizeof(FloatType);
  }
  else if (KernelTypeTrans::KERNEL_HALF == TYPE)
    this->kn_width_ = *reinterpret_cast<const TensorUInt *>(
        this->lib_loader_.dlsym("REG_SIZE")) / sizeof(FloatType) / 2;
  else
    this->kn_width_ = 1;
}


template <typename FloatType,
          KernelTypeTrans TYPE>
HPTC_INL void KernelTransProxy<FloatType, TYPE>::set_coef(
    const DeducedFloatType<FloatType> alpha,
    const DeducedFloatType<FloatType> beta) {
  this->set_reg_impl_(&this->reg_alpha_, alpha);
  this->set_reg_impl_(&this->reg_beta_, beta);
}


template <typename FloatType,
          KernelTypeTrans TYPE>
HPTC_INL void KernelTransProxy<FloatType, TYPE>::exec(
    const FloatType * RESTRICT input_data, FloatType * RESTRICT output_data,
    const TensorIdx input_stride, const TensorIdx output_stride) const {
  this->exec_impl_(input_data, output_data, input_stride, output_stride,
      &this->reg_alpha_, &this->reg_beta_);
}


template <typename FloatType,
          KernelTypeTrans TYPE>
HPTC_INL TensorUInt KernelTransProxy<FloatType, TYPE>::kn_width() const {
  return this->kn_width_;
}


template <typename FloatType,
          KernelTypeTrans TYPE>
template <typename Float,
          typename EnableType>
void KernelTransProxy<FloatType, TYPE>::init_func_ptr_(float) {
  if (KernelTypeTrans::KERNEL_FULL == TYPE) {
    this->set_reg_impl_ = reinterpret_cast<void (*)(void *, const float)>(
        this->lib_loader_.dlsym("set_trans_coef_full_s_"));
    this->exec_impl_ = reinterpret_cast<void (*)(const float *, float *,
        const TensorIdx, const TensorIdx, const void *, const void *)>(
        this->lib_loader_.dlsym("exec_trans_full_s_"));
  }
  else if (KernelTypeTrans::KERNEL_HALF == TYPE) {
    this->set_reg_impl_ = reinterpret_cast<void (*)(void *, const float)>(
        this->lib_loader_.dlsym("set_trans_coef_half_s_"));
    this->exec_impl_ = reinterpret_cast<void (*)(const float *, float *,
        const TensorIdx, const TensorIdx, const void *, const void *)>(
        this->lib_loader_.dlsym("exec_trans_half_s_"));
  }
  else {
    this->set_reg_impl_ = reinterpret_cast<void (*)(void *, const float)>(
        this->lib_loader_.dlsym("set_trans_coef_linear_s_"));
    this->exec_impl_ = reinterpret_cast<void (*)(const float *, float *,
        const TensorIdx, const TensorIdx, const void *, const void *)>(
        this->lib_loader_.dlsym("exec_trans_linear_s_"));
  }
}


template <typename FloatType,
          KernelTypeTrans TYPE>
template <typename Float,
          typename EnableType>
void KernelTransProxy<FloatType, TYPE>::init_func_ptr_(double) {
  if (KernelTypeTrans::KERNEL_FULL == TYPE) {
    this->set_reg_impl_ = reinterpret_cast<void (*)(void *, const double)>(
        this->lib_loader_.dlsym("set_trans_coef_full_s_"));
    this->exec_impl_ = reinterpret_cast<void (*)(const double *, double *,
        const TensorIdx, const TensorIdx, const void *, const void *)>(
        this->lib_loader_.dlsym("exec_trans_full_s_"));
  }
  else if (KernelTypeTrans::KERNEL_HALF == TYPE) {
    this->set_reg_impl_ = reinterpret_cast<void (*)(void *, const double)>(
        this->lib_loader_.dlsym("set_trans_coef_half_s_"));
    this->exec_impl_ = reinterpret_cast<void (*)(const double *, double *,
        const TensorIdx, const TensorIdx, const void *, const void *)>(
        this->lib_loader_.dlsym("exec_trans_half_s_"));
  }
  else {
    this->set_reg_impl_ = reinterpret_cast<void (*)(void *, const double)>(
        this->lib_loader_.dlsym("set_trans_coef_linear_s_"));
    this->exec_impl_ = reinterpret_cast<void (*)(const double *, double *,
        const TensorIdx, const TensorIdx, const void *, const void *)>(
        this->lib_loader_.dlsym("exec_trans_linear_s_"));
  }
}


template <typename FloatType,
          KernelTypeTrans TYPE>
template <typename Float,
          typename EnableType>
void KernelTransProxy<FloatType, TYPE>::init_func_ptr_(FloatComplex) {
  if (KernelTypeTrans::KERNEL_FULL == TYPE) {
    this->set_reg_impl_ = reinterpret_cast<void (*)(void *, const float)>(
        this->lib_loader_.dlsym("set_trans_coef_full_s_"));
    this->exec_impl_ = reinterpret_cast<void (*)(const FloatComplex *,
        FloatComplex *, const TensorIdx, const TensorIdx, const void *,
        const void *)>(this->lib_loader_.dlsym("exec_trans_full_s_"));
  }
  else if (KernelTypeTrans::KERNEL_HALF == TYPE) {
    this->set_reg_impl_ = reinterpret_cast<void (*)(void *, const float)>(
        this->lib_loader_.dlsym("set_trans_coef_half_s_"));
    this->exec_impl_ = reinterpret_cast<void (*)(const FloatComplex *,
        FloatComplex *, const TensorIdx, const TensorIdx, const void *,
        const void *)>(this->lib_loader_.dlsym("exec_trans_half_s_"));
  }
  else {
    this->set_reg_impl_ = reinterpret_cast<void (*)(void *, const float)>(
        this->lib_loader_.dlsym("set_trans_coef_linear_s_"));
    this->exec_impl_ = reinterpret_cast<void (*)(const FloatComplex *,
        FloatComplex *, const TensorIdx, const TensorIdx, const void *,
        const void *)>(this->lib_loader_.dlsym("exec_trans_linear_s_"));
  }
}


template <typename FloatType,
          KernelTypeTrans TYPE>
template <typename Float,
          typename EnableType>
void KernelTransProxy<FloatType, TYPE>::init_func_ptr_(DoubleComplex) {
  if (KernelTypeTrans::KERNEL_FULL == TYPE) {
    this->set_reg_impl_ = reinterpret_cast<void (*)(void *, const double)>(
        this->lib_loader_.dlsym("set_trans_coef_full_s_"));
    this->exec_impl_ = reinterpret_cast<void (*)(const DoubleComplex *,
        DoubleComplex *, const TensorIdx, const TensorIdx, const void *,
        const void *)>(this->lib_loader_.dlsym("exec_trans_full_s_"));
  }
  else if (KernelTypeTrans::KERNEL_HALF == TYPE) {
    this->set_reg_impl_ = reinterpret_cast<void (*)(void *, const double)>(
        this->lib_loader_.dlsym("set_trans_coef_half_s_"));
    this->exec_impl_ = reinterpret_cast<void (*)(const DoubleComplex *,
        DoubleComplex *, const TensorIdx, const TensorIdx, const void *,
        const void *)>(this->lib_loader_.dlsym("exec_trans_half_s_"));
  }
  else {
    this->set_reg_impl_ = reinterpret_cast<void (*)(void *, const double)>(
        this->lib_loader_.dlsym("set_trans_coef_linear_s_"));
    this->exec_impl_ = reinterpret_cast<void (*)(const DoubleComplex *,
        DoubleComplex *, const TensorIdx, const TensorIdx, const void *,
        const void *)>(this->lib_loader_.dlsym("exec_trans_linear_s_"));
  }
}


/*
 * Explicit template instantiation definition
 */
template class KernelTransProxy<float, KernelTypeTrans::KERNEL_FULL>;
template class KernelTransProxy<float, KernelTypeTrans::KERNEL_HALF>;
template class KernelTransProxy<float, KernelTypeTrans::KERNEL_LINE>;
template class KernelTransProxy<double, KernelTypeTrans::KERNEL_FULL>;
template class KernelTransProxy<double, KernelTypeTrans::KERNEL_HALF>;
template class KernelTransProxy<double, KernelTypeTrans::KERNEL_LINE>;
template class KernelTransProxy<FloatComplex, KernelTypeTrans::KERNEL_FULL>;
template class KernelTransProxy<FloatComplex, KernelTypeTrans::KERNEL_HALF>;
template class KernelTransProxy<FloatComplex, KernelTypeTrans::KERNEL_LINE>;
template class KernelTransProxy<DoubleComplex, KernelTypeTrans::KERNEL_FULL>;
template class KernelTransProxy<DoubleComplex, KernelTypeTrans::KERNEL_HALF>;
template class KernelTransProxy<DoubleComplex, KernelTypeTrans::KERNEL_LINE>;

}
