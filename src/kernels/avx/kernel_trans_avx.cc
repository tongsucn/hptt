#include <hptc/kernels/avx/kernel_trans_avx.h>

#include <immintrin.h>


namespace hptc {

INLINE DeducedRegType<float> reg_coef(float coef) {
  return _mm256_set1_ps(coef);
}


INLINE DeducedRegType<double> reg_coef(double coef) {
  return _mm256_set1_pd(coef);
}

}
