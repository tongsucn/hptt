#pragma once
#ifndef HPTC_KERNELS_ARCH_H_
#define HPTC_KERNELS_ARCH_H_

#include <immintrin.h>
#include <xmmintrin.h>

#include <string>

#include <hptc/types.h>
#include <hptc/util/util_trans.h>


namespace hptc {

union RegType {
  __m256  reg_avx2_fs;
  __m256d reg_avx2_fd;
  __m128  reg_avx2_hs;
  __m128d reg_avx2_hd;
  __m256  reg_avx_fs;
  __m256d reg_avx_fd;
  __m128  reg_avx_hs;
  __m128d reg_avx_hd;
  float   reg_common_fs;
  double  reg_common_fd;
  float   reg_common_hs;
  double  reg_common_hd;
};


class LibLoader {
public:
  // Delete copy/move constructors and operators
  LibLoader(const LibLoader &) = delete;
  LibLoader &operator=(const LibLoader &) = delete;
  LibLoader(LibLoader &&) = delete;
  LibLoader &operator=(LibLoader &&) = delete;

  static LibLoader &get_loader();
  void *dlsym(const char *symbol);

private:
  LibLoader();

  void *handler_;
};

}

#endif // HPTC_KERNELS_MICRO_KERNEL_ARCH_H_
