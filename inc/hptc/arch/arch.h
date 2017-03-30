#pragma once
#ifndef HPTC_ARCH_ARCH_H_
#define HPTC_ARCH_ARCH_H_

#include <immintrin.h>
#include <xmmintrin.h>

#include <vector>
#include <string>
#include <unordered_map>


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
  struct Arch_ {
    Arch_() = default;
    Arch_(bool found, const char *filename)
        : found(found), filename(filename) {}
    bool found;
    const std::string filename;
  };

  LibLoader();

  void init_cpu_();
  void init_path_();
  void select_arch_();
  void *search_(const std::string &filename);

  void *handler_;

  std::unordered_map<std::string, Arch_> intrin_sets_;
  std::vector<std::string> ld_list_;
};

}

#endif // HPTC_ARCH_ARCH_H_
