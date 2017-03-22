#ifndef STRANSPOSE_021_2144X64X384_PAR_H
#define STRANSPOSE_021_2144X64X384_PAR_H
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
#ifndef _TTC_STRANSPOSE8X4_0
#define _TTC_STRANSPOSE8X4_0
//B_ji = alpha * A_ij + beta * B_ji
template<int size0>
void sTranspose8x4_0(const float* __restrict__ A, int lda1, const int lda, float* __restrict__ B, const int ldb1, const int ldb  ,const float alpha ,const float beta)
{
   for(int ia = 0; ia < 8; ia++)
      for(int ib = 0; ib < 4; ib++)
         #pragma omp simd
         for(int i0 = 0; i0 < size0; i0++)
            B[i0 + ia * ldb + ib * ldb1] = alpha * A[i0 + ia * lda1 + ib * lda] + beta * B[i0 + ia * ldb + ib * ldb1];
}
#endif
/**
 * B(i0,i2,i1) <- alpha * A(i0,i1,i2) + beta * B(i0,i2,i1);
 */
template<int size0, int size1, int size2>
void sTranspose_021_2144x64x384_par( const float* __restrict__ A, float* __restrict__ B, const float alpha, const float beta, const int *lda, const int *ldb)
{
   int lda1;
   int lda2;
   if( lda == NULL ){
      lda1 = size0;
      lda2 = size1 * lda1;
   }else{
      lda1 = lda[0];
      lda2 = lda[1] * lda1;
   }
   int ldb1;
   int ldb2;
   if( ldb == NULL ){
      ldb1 = size0;
      ldb2 = size2 * ldb1;
   }else{
      ldb1 = ldb[0];
      ldb2 = ldb[1] * ldb1;
   }
   const int remainder1 = size1 % 8;
   const int remainder2 = size2 % 4;
#pragma omp parallel
   {
#pragma omp for collapse(2) schedule(static)
      for(int i1 = 0; i1 < size1 - 7; i1+= 8)
         for(int i2 = 0; i2 < size2 - 3; i2+= 4)
            sTranspose8x4_0<size0>(&A[i1*lda1 + i2*lda2], lda1, lda2, &B[i1*ldb2 + i2*ldb1], ldb1, ldb2  ,alpha ,beta);
      //Remainder loop
#pragma omp for collapse(2) schedule(static)
      for(int i1 = size1 - remainder1; i1 < size1; i1 += 1)
         for(int i2 = 0; i2 < size2 - remainder2; i2 += 1)
#pragma simd
            for(int i0 = 0; i0 < size0; i0++)
               B[i0 + i1*ldb2 + i2*ldb1] = alpha*A[i0 + i1*lda1 + i2*lda2] + beta*B[i0 + i1*ldb2 + i2*ldb1];
      //Remainder loop
#pragma omp for collapse(2) schedule(static)
      for(int i1 = 0; i1 < size1; i1 += 1)
         for(int i2 = size2 - remainder2; i2 < size2; i2 += 1)
#pragma simd
            for(int i0 = 0; i0 < size0; i0++)
               B[i0 + i1*ldb2 + i2*ldb1] = alpha*A[i0 + i1*lda1 + i2*lda2] + beta*B[i0 + i1*ldb2 + i2*ldb1];
   }
}
#endif
