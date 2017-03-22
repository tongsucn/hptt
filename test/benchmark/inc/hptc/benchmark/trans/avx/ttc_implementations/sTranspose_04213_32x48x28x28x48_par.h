#ifndef STRANSPOSE_04213_32X48X28X28X48_PAR_H
#define STRANSPOSE_04213_32X48X28X28X48_PAR_H
#include <xmmintrin.h>
#include <immintrin.h>
#include <complex.h>
#if defined(__ICC) || defined(__INTEL_COMPILER)
#define INLINE __forceinline
#elif defined(__GNUC__) || defined(__GNUG__)
#define INLINE __attribute__((always_inline))
#endif

#ifndef _TTC_STRANSPOSE1X1_0
#define _TTC_STRANSPOSE1X1_0
//B_ji = alpha * A_ij + beta * B_ji
template<int size0>
void sTranspose1x1_0(const float* __restrict__ A, int lda1, const int lda, float* __restrict__ B, const int ldb1, const int ldb  ,const float alpha ,const float beta)
{
   #pragma omp simd
   for(int i0 = 0; i0 < size0; i0++)
      B[i0] = alpha * A[i0] + beta * B[i0];
}
#endif
#ifndef _TTC_STRANSPOSE3X4_0
#define _TTC_STRANSPOSE3X4_0
//B_ji = alpha * A_ij + beta * B_ji
template<int size0>
void sTranspose3x4_0(const float* __restrict__ A, int lda1, const int lda, float* __restrict__ B, const int ldb1, const int ldb  ,const float alpha ,const float beta)
{
   for(int ia = 0; ia < 3; ia++)
      for(int ib = 0; ib < 4; ib++)
         #pragma omp simd
         for(int i0 = 0; i0 < size0; i0++)
            B[i0 + ia * ldb + ib * ldb1] = alpha * A[i0 + ia * lda1 + ib * lda] + beta * B[i0 + ia * ldb + ib * ldb1];
}
#endif
/**
 * B(i0,i4,i2,i1,i3) <- alpha * A(i0,i1,i2,i3,i4) + beta * B(i0,i4,i2,i1,i3);
 */
template<int size0, int size1, int size2, int size3, int size4>
void sTranspose_04213_32x48x28x28x48_par( const float* __restrict__ A, float* __restrict__ B, const float alpha, const float beta, const int *lda, const int *ldb)
{
   int lda1;
   int lda2;
   int lda3;
   int lda4;
   if( lda == NULL ){
      lda1 = size0;
      lda2 = size1 * lda1;
      lda3 = size2 * lda2;
      lda4 = size3 * lda3;
   }else{
      lda1 = lda[0];
      lda2 = lda[1] * lda1;
      lda3 = lda[2] * lda2;
      lda4 = lda[3] * lda3;
   }
   int ldb1;
   int ldb2;
   int ldb3;
   int ldb4;
   if( ldb == NULL ){
      ldb1 = size0;
      ldb2 = size4 * ldb1;
      ldb3 = size2 * ldb2;
      ldb4 = size1 * ldb3;
   }else{
      ldb1 = ldb[0];
      ldb2 = ldb[1] * ldb1;
      ldb3 = ldb[2] * ldb2;
      ldb4 = ldb[3] * ldb3;
   }
   const int remainder1 = size1 % 3;
   const int remainder4 = size4 % 4;
#pragma omp parallel
   {
#pragma omp for collapse(4) schedule(static)
      for(int i3 = 0; i3 < size3; i3+= 1)
         for(int i2 = 0; i2 < size2; i2+= 1)
            for(int i4 = 0; i4 < size4 - 3; i4+= 4)
               for(int i1 = 0; i1 < size1 - 2; i1+= 3)
                  sTranspose3x4_0<size0>(&A[i1*lda1 + i2*lda2 + i3*lda3 + i4*lda4], lda1, lda4, &B[i1*ldb3 + i2*ldb2 + i3*ldb4 + i4*ldb1], ldb1, ldb3  ,alpha ,beta);
      //Remainder loop
#pragma omp for collapse(4) schedule(static)
      for(int i3 = 0; i3 < size3; i3 += 1)
         for(int i2 = 0; i2 < size2; i2 += 1)
            for(int i4 = 0; i4 < size4 - remainder4; i4 += 1)
               for(int i1 = size1 - remainder1; i1 < size1; i1 += 1)
#pragma simd
                  for(int i0 = 0; i0 < size0; i0++)
                     B[i0 + i1*ldb3 + i2*ldb2 + i3*ldb4 + i4*ldb1] = alpha*A[i0 + i1*lda1 + i2*lda2 + i3*lda3 + i4*lda4] + beta*B[i0 + i1*ldb3 + i2*ldb2 + i3*ldb4 + i4*ldb1];
      //Remainder loop
#pragma omp for collapse(4) schedule(static)
      for(int i3 = 0; i3 < size3; i3 += 1)
         for(int i2 = 0; i2 < size2; i2 += 1)
            for(int i4 = size4 - remainder4; i4 < size4; i4 += 1)
               for(int i1 = 0; i1 < size1; i1 += 1)
#pragma simd
                  for(int i0 = 0; i0 < size0; i0++)
                     B[i0 + i1*ldb3 + i2*ldb2 + i3*ldb4 + i4*ldb1] = alpha*A[i0 + i1*lda1 + i2*lda2 + i3*lda3 + i4*lda4] + beta*B[i0 + i1*ldb3 + i2*ldb2 + i3*ldb4 + i4*ldb1];
   }
}
#endif
