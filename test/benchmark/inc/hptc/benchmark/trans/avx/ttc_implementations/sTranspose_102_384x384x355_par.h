#ifndef STRANSPOSE_102_384X384X355_PAR_H
#define STRANSPOSE_102_384X384X355_PAR_H
#include <xmmintrin.h>
#include <immintrin.h>
#include <complex.h>
#if defined(__ICC) || defined(__INTEL_COMPILER)
#define INLINE __forceinline
#elif defined(__GNUC__) || defined(__GNUG__)
#define INLINE __attribute__((always_inline))
#endif

#include <queue>
#include "ttc_offset.h"
#ifndef _TTC_STRANSPOSE8X8
#define _TTC_STRANSPOSE8X8
//B_ji = alpha * A_ij + beta * B_ji
static INLINE void sTranspose8x8(const float* __restrict__ A, const int lda, float* __restrict__ B, const int ldb  ,const __m256 &reg_alpha ,const __m256 &reg_beta)
{
   //Load A
   __m256 rowA0 = _mm256_load_ps((A + 0 +0*lda));
   __m256 rowA1 = _mm256_load_ps((A + 0 +1*lda));
   __m256 rowA2 = _mm256_load_ps((A + 0 +2*lda));
   __m256 rowA3 = _mm256_load_ps((A + 0 +3*lda));
   __m256 rowA4 = _mm256_load_ps((A + 0 +4*lda));
   __m256 rowA5 = _mm256_load_ps((A + 0 +5*lda));
   __m256 rowA6 = _mm256_load_ps((A + 0 +6*lda));
   __m256 rowA7 = _mm256_load_ps((A + 0 +7*lda));

   //8x8 transpose micro kernel
   __m256 r121, r139, r120, r138, r71, r89, r70, r88, r11, r1, r55, r29, r10, r0, r54, r28;
   r28 = _mm256_unpacklo_ps( rowA4, rowA5 );
   r54 = _mm256_unpacklo_ps( rowA6, rowA7 );
    r0 = _mm256_unpacklo_ps( rowA0, rowA1 );
   r10 = _mm256_unpacklo_ps( rowA2, rowA3 );
   r29 = _mm256_unpackhi_ps( rowA4, rowA5 );
   r55 = _mm256_unpackhi_ps( rowA6, rowA7 );
    r1 = _mm256_unpackhi_ps( rowA0, rowA1 );
   r11 = _mm256_unpackhi_ps( rowA2, rowA3 );
   r88 = _mm256_shuffle_ps( r28, r54, 0x44 );
   r70 = _mm256_shuffle_ps( r0, r10, 0x44 );
   r89 = _mm256_shuffle_ps( r28, r54, 0xee );
   r71 = _mm256_shuffle_ps( r0, r10, 0xee );
   r138 = _mm256_shuffle_ps( r29, r55, 0x44 );
   r120 = _mm256_shuffle_ps( r1, r11, 0x44 );
   r139 = _mm256_shuffle_ps( r29, r55, 0xee );
   r121 = _mm256_shuffle_ps( r1, r11, 0xee );
   rowA0 = _mm256_permute2f128_ps( r88, r70, 0x2 );
   rowA1 = _mm256_permute2f128_ps( r89, r71, 0x2 );
   rowA2 = _mm256_permute2f128_ps( r138, r120, 0x2 );
   rowA3 = _mm256_permute2f128_ps( r139, r121, 0x2 );
   rowA4 = _mm256_permute2f128_ps( r88, r70, 0x13 );
   rowA5 = _mm256_permute2f128_ps( r89, r71, 0x13 );
   rowA6 = _mm256_permute2f128_ps( r138, r120, 0x13 );
   rowA7 = _mm256_permute2f128_ps( r139, r121, 0x13 );

   //Scale A
   rowA0 = _mm256_mul_ps(rowA0, reg_alpha);
   rowA1 = _mm256_mul_ps(rowA1, reg_alpha);
   rowA2 = _mm256_mul_ps(rowA2, reg_alpha);
   rowA3 = _mm256_mul_ps(rowA3, reg_alpha);
   rowA4 = _mm256_mul_ps(rowA4, reg_alpha);
   rowA5 = _mm256_mul_ps(rowA5, reg_alpha);
   rowA6 = _mm256_mul_ps(rowA6, reg_alpha);
   rowA7 = _mm256_mul_ps(rowA7, reg_alpha);

   //Load B
   __m256 rowB0 = _mm256_load_ps((B + 0 +0*ldb));
   __m256 rowB1 = _mm256_load_ps((B + 0 +1*ldb));
   __m256 rowB2 = _mm256_load_ps((B + 0 +2*ldb));
   __m256 rowB3 = _mm256_load_ps((B + 0 +3*ldb));
   __m256 rowB4 = _mm256_load_ps((B + 0 +4*ldb));
   __m256 rowB5 = _mm256_load_ps((B + 0 +5*ldb));
   __m256 rowB6 = _mm256_load_ps((B + 0 +6*ldb));
   __m256 rowB7 = _mm256_load_ps((B + 0 +7*ldb));

   rowB0 = _mm256_add_ps( _mm256_mul_ps(rowB0, reg_beta), rowA0);
   rowB1 = _mm256_add_ps( _mm256_mul_ps(rowB1, reg_beta), rowA1);
   rowB2 = _mm256_add_ps( _mm256_mul_ps(rowB2, reg_beta), rowA2);
   rowB3 = _mm256_add_ps( _mm256_mul_ps(rowB3, reg_beta), rowA3);
   rowB4 = _mm256_add_ps( _mm256_mul_ps(rowB4, reg_beta), rowA4);
   rowB5 = _mm256_add_ps( _mm256_mul_ps(rowB5, reg_beta), rowA5);
   rowB6 = _mm256_add_ps( _mm256_mul_ps(rowB6, reg_beta), rowA6);
   rowB7 = _mm256_add_ps( _mm256_mul_ps(rowB7, reg_beta), rowA7);
   //Store B
   _mm256_store_ps((B + 0 + 0 * ldb), rowB0);
   _mm256_store_ps((B + 0 + 1 * ldb), rowB1);
   _mm256_store_ps((B + 0 + 2 * ldb), rowB2);
   _mm256_store_ps((B + 0 + 3 * ldb), rowB3);
   _mm256_store_ps((B + 0 + 4 * ldb), rowB4);
   _mm256_store_ps((B + 0 + 5 * ldb), rowB5);
   _mm256_store_ps((B + 0 + 6 * ldb), rowB6);
   _mm256_store_ps((B + 0 + 7 * ldb), rowB7);
}
#endif
#ifndef _TTC_STRANSPOSE32X16
#define _TTC_STRANSPOSE32X16
//B_ji = alpha * A_ij + beta * B_ji
static INLINE void sTranspose32x16(const float* __restrict__ A, const int lda, float* __restrict__ B, const int ldb  ,const __m256 &reg_alpha ,const __m256 &reg_beta)
{
   //invoke micro-transpose
   sTranspose8x8(A, lda, B, ldb  , reg_alpha , reg_beta);

   //invoke micro-transpose
   sTranspose8x8(A + 8 * lda, lda, B + 8, ldb  , reg_alpha , reg_beta);

   //invoke micro-transpose
   sTranspose8x8(A + 8, lda, B + 8 * ldb, ldb  , reg_alpha , reg_beta);

   //invoke micro-transpose
   sTranspose8x8(A + 8 + 8 * lda, lda, B + 8 + 8 * ldb, ldb  , reg_alpha , reg_beta);

   //invoke micro-transpose
   sTranspose8x8(A + 16, lda, B + 16 * ldb, ldb  , reg_alpha , reg_beta);

   //invoke micro-transpose
   sTranspose8x8(A + 16 + 8 * lda, lda, B + 8 + 16 * ldb, ldb  , reg_alpha , reg_beta);

   //invoke micro-transpose
   sTranspose8x8(A + 24, lda, B + 24 * ldb, ldb  , reg_alpha , reg_beta);

   //invoke micro-transpose
   sTranspose8x8(A + 24 + 8 * lda, lda, B + 8 + 24 * ldb, ldb  , reg_alpha , reg_beta);

}
#endif
#ifndef _TTC_STRANSPOSE32X16_PREFETCH_5
#define _TTC_STRANSPOSE32X16_PREFETCH_5
//B_ji = alpha * A_ij + beta * B_ji
static INLINE void sTranspose32x16_prefetch_5(const float* __restrict__ A, const int lda, float* __restrict__ B, const int ldb, const float* __restrict__ Anext0, float* __restrict__ Bnext0, const float* __restrict__ Anext1, float* __restrict__ Bnext1  ,const __m256 &reg_alpha ,const __m256 &reg_beta)
{
   //prefetch A
   _mm_prefetch((char*)(Anext0 + 16 + 8 * lda), _MM_HINT_T2);
   _mm_prefetch((char*)(Anext0 + 16 + 9 * lda), _MM_HINT_T2);
   _mm_prefetch((char*)(Anext0 + 16 + 10 * lda), _MM_HINT_T2);
   _mm_prefetch((char*)(Anext0 + 16 + 11 * lda), _MM_HINT_T2);
   _mm_prefetch((char*)(Anext0 + 16 + 12 * lda), _MM_HINT_T2);
   _mm_prefetch((char*)(Anext0 + 16 + 13 * lda), _MM_HINT_T2);
   _mm_prefetch((char*)(Anext0 + 16 + 14 * lda), _MM_HINT_T2);
   _mm_prefetch((char*)(Anext0 + 16 + 15 * lda), _MM_HINT_T2);
   //invoke micro-transpose
   sTranspose8x8(A, lda, B, ldb  , reg_alpha , reg_beta);

   //prefetch B
   _mm_prefetch((char*)(Bnext0 + 24 * ldb), _MM_HINT_T2);
   _mm_prefetch((char*)(Bnext0 + 25 * ldb), _MM_HINT_T2);
   _mm_prefetch((char*)(Bnext0 + 26 * ldb), _MM_HINT_T2);
   _mm_prefetch((char*)(Bnext0 + 27 * ldb), _MM_HINT_T2);
   _mm_prefetch((char*)(Bnext0 + 28 * ldb), _MM_HINT_T2);
   _mm_prefetch((char*)(Bnext0 + 29 * ldb), _MM_HINT_T2);
   _mm_prefetch((char*)(Bnext0 + 30 * ldb), _MM_HINT_T2);
   _mm_prefetch((char*)(Bnext0 + 31 * ldb), _MM_HINT_T2);
   //invoke micro-transpose
   sTranspose8x8(A + 8 * lda, lda, B + 8, ldb  , reg_alpha , reg_beta);

   //invoke micro-transpose
   sTranspose8x8(A + 8, lda, B + 8 * ldb, ldb  , reg_alpha , reg_beta);

   //prefetch A
   _mm_prefetch((char*)(Anext1 + 0 * lda), _MM_HINT_T2);
   _mm_prefetch((char*)(Anext1 + 1 * lda), _MM_HINT_T2);
   _mm_prefetch((char*)(Anext1 + 2 * lda), _MM_HINT_T2);
   _mm_prefetch((char*)(Anext1 + 3 * lda), _MM_HINT_T2);
   _mm_prefetch((char*)(Anext1 + 4 * lda), _MM_HINT_T2);
   _mm_prefetch((char*)(Anext1 + 5 * lda), _MM_HINT_T2);
   _mm_prefetch((char*)(Anext1 + 6 * lda), _MM_HINT_T2);
   _mm_prefetch((char*)(Anext1 + 7 * lda), _MM_HINT_T2);
   //prefetch B
   _mm_prefetch((char*)(Bnext1 + 0 * ldb), _MM_HINT_T2);
   _mm_prefetch((char*)(Bnext1 + 1 * ldb), _MM_HINT_T2);
   _mm_prefetch((char*)(Bnext1 + 2 * ldb), _MM_HINT_T2);
   _mm_prefetch((char*)(Bnext1 + 3 * ldb), _MM_HINT_T2);
   _mm_prefetch((char*)(Bnext1 + 4 * ldb), _MM_HINT_T2);
   _mm_prefetch((char*)(Bnext1 + 5 * ldb), _MM_HINT_T2);
   _mm_prefetch((char*)(Bnext1 + 6 * ldb), _MM_HINT_T2);
   _mm_prefetch((char*)(Bnext1 + 7 * ldb), _MM_HINT_T2);
   //invoke micro-transpose
   sTranspose8x8(A + 8 + 8 * lda, lda, B + 8 + 8 * ldb, ldb  , reg_alpha , reg_beta);

   //prefetch A
   _mm_prefetch((char*)(Anext1 + 8 * lda), _MM_HINT_T2);
   _mm_prefetch((char*)(Anext1 + 9 * lda), _MM_HINT_T2);
   _mm_prefetch((char*)(Anext1 + 10 * lda), _MM_HINT_T2);
   _mm_prefetch((char*)(Anext1 + 11 * lda), _MM_HINT_T2);
   _mm_prefetch((char*)(Anext1 + 12 * lda), _MM_HINT_T2);
   _mm_prefetch((char*)(Anext1 + 13 * lda), _MM_HINT_T2);
   _mm_prefetch((char*)(Anext1 + 14 * lda), _MM_HINT_T2);
   _mm_prefetch((char*)(Anext1 + 15 * lda), _MM_HINT_T2);
   //invoke micro-transpose
   sTranspose8x8(A + 16, lda, B + 16 * ldb, ldb  , reg_alpha , reg_beta);

   //prefetch B
   _mm_prefetch((char*)(Bnext1 + 8 * ldb), _MM_HINT_T2);
   _mm_prefetch((char*)(Bnext1 + 9 * ldb), _MM_HINT_T2);
   _mm_prefetch((char*)(Bnext1 + 10 * ldb), _MM_HINT_T2);
   _mm_prefetch((char*)(Bnext1 + 11 * ldb), _MM_HINT_T2);
   _mm_prefetch((char*)(Bnext1 + 12 * ldb), _MM_HINT_T2);
   _mm_prefetch((char*)(Bnext1 + 13 * ldb), _MM_HINT_T2);
   _mm_prefetch((char*)(Bnext1 + 14 * ldb), _MM_HINT_T2);
   _mm_prefetch((char*)(Bnext1 + 15 * ldb), _MM_HINT_T2);
   //invoke micro-transpose
   sTranspose8x8(A + 16 + 8 * lda, lda, B + 8 + 16 * ldb, ldb  , reg_alpha , reg_beta);

   //invoke micro-transpose
   sTranspose8x8(A + 24, lda, B + 24 * ldb, ldb  , reg_alpha , reg_beta);

   //prefetch A
   _mm_prefetch((char*)(Anext1 + 16 + 0 * lda), _MM_HINT_T2);
   _mm_prefetch((char*)(Anext1 + 16 + 1 * lda), _MM_HINT_T2);
   _mm_prefetch((char*)(Anext1 + 16 + 2 * lda), _MM_HINT_T2);
   _mm_prefetch((char*)(Anext1 + 16 + 3 * lda), _MM_HINT_T2);
   _mm_prefetch((char*)(Anext1 + 16 + 4 * lda), _MM_HINT_T2);
   _mm_prefetch((char*)(Anext1 + 16 + 5 * lda), _MM_HINT_T2);
   _mm_prefetch((char*)(Anext1 + 16 + 6 * lda), _MM_HINT_T2);
   _mm_prefetch((char*)(Anext1 + 16 + 7 * lda), _MM_HINT_T2);
   //prefetch B
   _mm_prefetch((char*)(Bnext1 + 16 * ldb), _MM_HINT_T2);
   _mm_prefetch((char*)(Bnext1 + 17 * ldb), _MM_HINT_T2);
   _mm_prefetch((char*)(Bnext1 + 18 * ldb), _MM_HINT_T2);
   _mm_prefetch((char*)(Bnext1 + 19 * ldb), _MM_HINT_T2);
   _mm_prefetch((char*)(Bnext1 + 20 * ldb), _MM_HINT_T2);
   _mm_prefetch((char*)(Bnext1 + 21 * ldb), _MM_HINT_T2);
   _mm_prefetch((char*)(Bnext1 + 22 * ldb), _MM_HINT_T2);
   _mm_prefetch((char*)(Bnext1 + 23 * ldb), _MM_HINT_T2);
   //invoke micro-transpose
   sTranspose8x8(A + 24 + 8 * lda, lda, B + 8 + 24 * ldb, ldb  , reg_alpha , reg_beta);

}
#endif
/**
 * B(i1,i0,i2) <- alpha * A(i0,i1,i2) + beta * B(i1,i0,i2);
 */
template<int size0, int size1, int size2>
void sTranspose_102_384x384x355_par( const float* __restrict__ A, float* __restrict__ B, const float alpha, const float beta, const int *lda, const int *ldb)
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
      ldb1 = size1;
      ldb2 = size0 * ldb1;
   }else{
      ldb1 = ldb[0];
      ldb2 = ldb[1] * ldb1;
   }
   const int remainder0 = size0 % 32;
   const int remainder1 = size1 % 16;
   //broadcast reg_alpha
   __m256 reg_alpha = _mm256_set1_ps(alpha);
   //broadcast reg_beta
   __m256 reg_beta = _mm256_set1_ps(beta);
#pragma omp parallel
   {
      int counter = 0;
      std::queue<Offset> tasks;
#pragma omp for collapse(3) schedule(static)
      for(int i0 = 0; i0 < size0 - 31; i0+= 32)
         for(int i1 = 0; i1 < size1 - 15; i1+= 16)
            for(int i2 = 0; i2 < size2; i2+= 1)
            {
               int offsetA = i0 + i1*lda1 + i2*lda2;
               int offsetB = i0*ldb1 + i1 + i2*ldb2;
               if( counter >= 1 ){
                  const Offset &task = tasks.back();
                  int offsetAnext0 = task.offsetA;
                  int offsetBnext0 = task.offsetB;
                  const Offset &currentTask = tasks.front();
                  sTranspose32x16_prefetch_5(&A[currentTask.offsetA], lda1, &B[currentTask.offsetB], ldb1, &A[offsetAnext0], &B[offsetBnext0], &A[offsetA], &B[offsetB]  ,reg_alpha ,reg_beta);
                  tasks.pop();
               }
               counter++;
               Offset offset; offset.offsetA = offsetA; offset.offsetB = offsetB;
               tasks.push( offset );
            }
      while(tasks.size() > 0){
          const Offset &task = tasks.front();
          sTranspose32x16(&A[task.offsetA], lda1, &B[task.offsetB], ldb1   ,reg_alpha ,reg_beta);
          tasks.pop();
      }
      //Remainder loop
#pragma omp for collapse(2) schedule(static)
      for(int i0 = size0 - remainder0; i0 < size0; i0 += 1)
         for(int i1 = 0; i1 < size1 - remainder1; i1 += 1)
            for(int i2 = 0; i2 < size2; i2 += 1)
               B[i0*ldb1 + i1 + i2*ldb2] = alpha*A[i0 + i1*lda1 + i2*lda2] + beta*B[i0*ldb1 + i1 + i2*ldb2];
      //Remainder loop
#pragma omp for collapse(2) schedule(static)
      for(int i0 = 0; i0 < size0; i0 += 1)
         for(int i1 = size1 - remainder1; i1 < size1; i1 += 1)
            for(int i2 = 0; i2 < size2; i2 += 1)
               B[i0*ldb1 + i1 + i2*ldb2] = alpha*A[i0 + i1*lda1 + i2*lda2] + beta*B[i0*ldb1 + i1 + i2*ldb2];
   }
}
#endif
