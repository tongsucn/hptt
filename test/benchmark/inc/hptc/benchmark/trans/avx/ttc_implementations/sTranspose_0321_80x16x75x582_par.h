#ifndef STRANSPOSE_0321_80X16X75X582_PAR_H
#define STRANSPOSE_0321_80X16X75X582_PAR_H
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
#ifndef _TTC_STRANSPOSE8X1_0
#define _TTC_STRANSPOSE8X1_0
//B_ji = alpha * A_ij + beta * B_ji
template<int size0>
void sTranspose8x1_0(const float* __restrict__ A, int lda1, const int lda, float* __restrict__ B, const int ldb1, const int ldb  ,const float alpha ,const float beta)
{
   for(int ia = 0; ia < 8; ia++)
      #pragma omp simd
      for(int i0 = 0; i0 < size0; i0++)
         B[i0 + ia * ldb] = alpha * A[i0 + ia * lda1] + beta * B[i0 + ia * ldb];
}
#endif
/**
 * B(i0,i3,i2,i1) <- alpha * A(i0,i1,i2,i3) + beta * B(i0,i3,i2,i1);
 */
template<int size0, int size1, int size2, int size3>
void sTranspose_0321_80x16x75x582_par( const float* __restrict__ A, float* __restrict__ B, const float alpha, const float beta, const int *lda, const int *ldb)
{
   int lda1;
   int lda2;
   int lda3;
   if( lda == NULL ){
      lda1 = size0;
      lda2 = size1 * lda1;
      lda3 = size2 * lda2;
   }else{
      lda1 = lda[0];
      lda2 = lda[1] * lda1;
      lda3 = lda[2] * lda2;
   }
   int ldb1;
   int ldb2;
   int ldb3;
   if( ldb == NULL ){
      ldb1 = size0;
      ldb2 = size3 * ldb1;
      ldb3 = size2 * ldb2;
   }else{
      ldb1 = ldb[0];
      ldb2 = ldb[1] * ldb1;
      ldb3 = ldb[2] * ldb2;
   }
   const int remainder1 = size1 % 8;
   const int remainder3 = size3 % 1;
#pragma omp parallel
   {
#pragma omp for collapse(3) schedule(static)
      for(int i3 = 0; i3 < size3; i3+= 1)
         for(int i2 = 0; i2 < size2; i2+= 1)
            for(int i1 = 0; i1 < size1 - 7; i1+= 8)
               sTranspose8x1_0<size0>(&A[i1*lda1 + i2*lda2 + i3*lda3], lda1, lda3, &B[i1*ldb3 + i2*ldb2 + i3*ldb1], ldb1, ldb3  ,alpha ,beta);
      //Remainder loop
#pragma omp for collapse(3) schedule(static)
      for(int i3 = 0; i3 < size3 - remainder3; i3 += 1)
         for(int i2 = 0; i2 < size2; i2 += 1)
            for(int i1 = size1 - remainder1; i1 < size1; i1 += 1)
#pragma simd
               for(int i0 = 0; i0 < size0; i0++)
                  B[i0 + i1*ldb3 + i2*ldb2 + i3*ldb1] = alpha*A[i0 + i1*lda1 + i2*lda2 + i3*lda3] + beta*B[i0 + i1*ldb3 + i2*ldb2 + i3*ldb1];
      //Remainder loop
#pragma omp for collapse(3) schedule(static)
      for(int i3 = size3 - remainder3; i3 < size3; i3 += 1)
         for(int i2 = 0; i2 < size2; i2 += 1)
            for(int i1 = 0; i1 < size1; i1 += 1)
#pragma simd
               for(int i0 = 0; i0 < size0; i0++)
                  B[i0 + i1*ldb3 + i2*ldb2 + i3*ldb1] = alpha*A[i0 + i1*lda1 + i2*lda2 + i3*lda3] + beta*B[i0 + i1*ldb3 + i2*ldb2 + i3*ldb1];
   }
}
#endif
