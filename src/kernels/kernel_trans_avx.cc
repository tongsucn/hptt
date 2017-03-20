#include <hptc/kernels/avx/kernel_trans_avx.h>

#include <xmmintrin.h>
#include <immintrin.h>

#include <type_traits>

#include <hptc/types.h>
#include <hptc/config/config_trans.h>


namespace hptc {

/*
 * Implementation for class KernelTransAvxBase
 */
template <typename FloatType,
          KernelTypeTrans TYPE>
KernelTransAvxBase<FloatType, TYPE>::KernelTransAvxBase(Deduced coef_alpha,
    Deduced coef_beta)
    : reg_alpha(this->reg_coef(coef_alpha)),
      reg_beta(this->reg_coef(coef_beta)) {
}


template <typename FloatType,
          KernelTypeTrans TYPE>
GenNumType KernelTransAvxBase<FloatType, TYPE>::get_kernel_width() const {
  constexpr GenNumType width = REG_SIZE_BYTE_AVX / sizeof(FloatType);
  return TYPE == KernelTypeTrans::KERNEL_HALF ? width / 2 : width;
}


template <typename FloatType,
          KernelTypeTrans TYPE>
GenNumType KernelTransAvxBase<FloatType, TYPE>::get_reg_num() const {
  return this->get_kernel_width();
}


template <typename FloatType,
          KernelTypeTrans TYPE>
template <KernelTypeTrans KERNEL,
          std::enable_if_t<KERNEL == KernelTypeTrans::KERNEL_FULL or
              KERNEL == KernelTypeTrans::KERNEL_LINE> *>
DeducedRegType<float, KERNEL>
KernelTransAvxBase<FloatType, TYPE>::reg_coef(float coef) const {
  return _mm256_set1_ps(coef);
}


template <typename FloatType,
          KernelTypeTrans TYPE>
template <KernelTypeTrans KERNEL,
          std::enable_if_t<KERNEL == KernelTypeTrans::KERNEL_FULL or
              KERNEL == KernelTypeTrans::KERNEL_LINE> *>
DeducedRegType<double, KERNEL>
KernelTransAvxBase<FloatType, TYPE>::reg_coef(double coef) const {
  return _mm256_set1_pd(coef);
}


template <typename FloatType,
          KernelTypeTrans TYPE>
template <KernelTypeTrans KERNEL,
          std::enable_if_t<KERNEL == KernelTypeTrans::KERNEL_HALF> *>
DeducedRegType<float, KERNEL>
KernelTransAvxBase<FloatType, TYPE>::reg_coef(float coef) const {
  return _mm_set1_ps(coef);
}


template <typename FloatType,
          KernelTypeTrans TYPE>
template <KernelTypeTrans KERNEL,
          std::enable_if_t<std::is_same<FloatType, double>::value and
              KERNEL == KernelTypeTrans::KERNEL_HALF> *>
DeducedRegType<double, KERNEL>
KernelTransAvxBase<FloatType, TYPE>::reg_coef(double coef) const {
  return _mm_set1_pd(coef);
}


template <typename FloatType,
          KernelTypeTrans TYPE>
template <KernelTypeTrans KERNEL,
          std::enable_if_t<std::is_same<FloatType, DoubleComplex>::value and
              KERNEL == KernelTypeTrans::KERNEL_HALF> *>
DeducedRegType<DoubleComplex, KERNEL>
KernelTransAvxBase<FloatType, TYPE>::reg_coef(double coef) const {
  return coef;
}


/*
 * Explicit instantiation for struct KernelTransAvxBase
 */
template struct KernelTransAvxBase<float, KernelTypeTrans::KERNEL_FULL>;
template struct KernelTransAvxBase<double, KernelTypeTrans::KERNEL_FULL>;
template struct KernelTransAvxBase<FloatComplex, KernelTypeTrans::KERNEL_FULL>;
template struct KernelTransAvxBase<DoubleComplex, KernelTypeTrans::KERNEL_FULL>;
template struct KernelTransAvxBase<float, KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransAvxBase<double, KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransAvxBase<FloatComplex, KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransAvxBase<DoubleComplex, KernelTypeTrans::KERNEL_HALF>;
template struct KernelTransAvxBase<float, KernelTypeTrans::KERNEL_LINE>;
template struct KernelTransAvxBase<double, KernelTypeTrans::KERNEL_LINE>;
template struct KernelTransAvxBase<FloatComplex, KernelTypeTrans::KERNEL_LINE>;
template struct KernelTransAvxBase<DoubleComplex, KernelTypeTrans::KERNEL_LINE>;

}
