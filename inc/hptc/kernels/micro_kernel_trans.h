#pragma once
#ifndef HPTC_KERNELS_MICRO_KERNEL_TRANS_H_
#define HPTC_KERNELS_MICRO_KERNEL_TRANS_H_

#include <type_traits>

#include <hptc/util/util.h>
#include <hptc/arch/arch.h>
#include <hptc/util/util_trans.h>


namespace hptc {

class KernelTransProxyBase {
public:
  KernelTransProxyBase();

protected:
  LibLoader &lib_loader_;
  RegType reg_alpha_, reg_beta_;
};


template <typename FloatType,
          KernelTypeTrans TYPE>
class KernelTransProxy : public KernelTransProxyBase {
public:
  using Float = FloatType;

  KernelTransProxy();

  void set_coef(const DeducedFloatType<FloatType> alpha,
      const DeducedFloatType<FloatType> beta);

  void exec(const FloatType * RESTRICT input_data,
      FloatType * RESTRICT output_data, const TensorIdx input_stride,
      const TensorIdx output_stride) const;

  TensorUInt kn_width() const;

private:
  template <typename Float,
            typename EnableType = Enable<
                std::is_same<float, Float>::value, float>>
  void init_func_ptr_(float);
  template <typename Float,
            typename EnableType = Enable<
                std::is_same<double, Float>::value, double>>
  void init_func_ptr_(double);
  template <typename Float,
            typename EnableType = Enable<
                std::is_same<FloatComplex, Float>::value, FloatComplex>>
  void init_func_ptr_(FloatComplex);
  template <typename Float,
            typename EnableType = Enable<
                std::is_same<DoubleComplex, Float>::value, DoubleComplex>>
  void init_func_ptr_(DoubleComplex);

  void (*set_reg_impl_)(void *reg, const DeducedFloatType<FloatType>);
  void (*exec_impl_)(const FloatType *, FloatType *, const TensorIdx,
      const TensorIdx, const void *, const void *);

  TensorUInt kn_width_;
};


/*
 * Alias for micro kernels
 */
template <typename FloatType>
using KernelTransFull = KernelTransProxy<FloatType,
    KernelTypeTrans::KERNEL_FULL>;


template <typename FloatType>
using KernelTransHalf = KernelTransProxy<FloatType,
    KernelTypeTrans::KERNEL_HALF>;


template <typename FloatType>
using KernelTransLinear = KernelTransProxy<FloatType,
    KernelTypeTrans::KERNEL_LINE>;


/*
 * Improve explicit template instantiation declaration of class KernelTransProxy
 */
#include "micro_kernel_trans.tcc"

}

#endif // HPTC_KERNELS_MICRO_KERNEL_TRANS_H_
