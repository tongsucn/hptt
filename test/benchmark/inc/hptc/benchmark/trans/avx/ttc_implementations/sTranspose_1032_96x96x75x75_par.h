#ifndef STRANSPOSE_1032_96X96X75X75_PAR_H
#define STRANSPOSE_1032_96X96X75X75_PAR_H
#include <xmmintrin.h>
#include <immintrin.h>
#include <complex.h>
/**
 * B(i1,i0,i3,i2) <- alpha * A(i0,i1,i2,i3) + beta * B(i1,i0,i3,i2);
 */
template<int size0, int size1, int size2, int size3>
void sTranspose_1032_96x96x75x75_par( const float* __restrict__ A, float* __restrict__ B, const float alpha, const float beta, const int *lda, const int *ldb)
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
      ldb1 = size1;
      ldb2 = size0 * ldb1;
      ldb3 = size3 * ldb2;
   }else{
      ldb1 = ldb[0];
      ldb2 = ldb[1] * ldb1;
      ldb3 = ldb[2] * ldb2;
   }
   const int remainder0 = size0 % 1;
   const int remainder1 = size1 % 1;
#pragma omp parallel for collapse(3)
   for(int i2 = 0; i2 < size2; i2 += 1)
      for(int i3 = 0; i3 < size3; i3 += 1)
         for(int i0 = 0; i0 < size0; i0 += 1)
#pragma simd
            for(int i1 = 0; i1 < size1; i1 += 1)
               B[i0*ldb1 + i1 + i2*ldb3 + i3*ldb2] = alpha*A[i0 + i1*lda1 + i2*lda2 + i3*lda3] + beta*B[i0*ldb1 + i1 + i2*ldb3 + i3*ldb2];
}
#endif
