#include <hptc/arch/arch.h>

#include <cstdint>
#include <cstdlib>

#include <cpuid.h>
#include <dlfcn.h>

#include <string>
#include <sstream>

#include <hptc/arch/compat.h>


namespace hptc {

void hptc_cpuid(const uint32_t input, uint32_t output[4]) {
  __cpuid_count(input, 0, output[0], output[1], output[2], output[3]);
}


LibLoader &LibLoader::get_loader() {
  static LibLoader loader;
  return loader;
}


void *LibLoader::dlsym(const char *symbol) {
  return ::dlsym(this->handler_, symbol);
}


LibLoader::LibLoader()
    : handler_(nullptr),
      intrin_sets_{ { "avx2", Arch_(false, "libhptc_avx2.so") },
          { "avx", Arch_(false, "libhptc_avx.so") },
          { "fma3", Arch_(false, "libhptc_fma3.so") },
          { "common", Arch_(true, "libhptc_common.so") } },
      ld_list_{ "" } {
  // Detect CPU features
  this->init_cpu_();

  // Parse environment variable LD_LIBRARY_PATH
  this->init_path_();

  // Select and load shared library
  this->select_arch_();
}


void LibLoader::init_cpu_() {
  // CPU detection is from https://github.com/Mysticial/FeatureDetector
  uint32_t cpu_info[4];

  // Get support values
  hptc::hptc_cpuid(0, cpu_info);
  const auto support_val = cpu_info[0];
  hptc::hptc_cpuid(0x80000000, cpu_info);

  // Check supported features
  if (support_val >= 1) {
    hptc::hptc_cpuid(1, cpu_info);

    // Check AVX
    this->intrin_sets_["avx"].found = 0 != (cpu_info[2] & (1 << 28));

    // Check FMA3
    this->intrin_sets_["fma3"].found = 0 != (cpu_info[2] & (1 << 12));
  }

  if (support_val >= 7) {
    hptc::hptc_cpuid(7, cpu_info);

    // Check AVX2
    this->intrin_sets_["avx2"].found = 0 != (cpu_info[1] & (1 << 5));
  }
}


void LibLoader::init_path_() {
  std::stringstream ld_env_str_stream;
  ld_env_str_stream << std::getenv("LD_LIBRARY_PATH");

  while (std::getline(ld_env_str_stream, this->ld_list_.back(), ':')) {
    this->ld_list_.back() += 0 == this->ld_list_.back().length() ? "./" : "/";
    this->ld_list_.emplace_back();
  }
  this->ld_list_.back() = "./";
}


void LibLoader::select_arch_() {
  if (nullptr == this->handler_ and this->intrin_sets_["avx2"].found)
    this->handler_ = this->search_(this->intrin_sets_["avx2"].filename);
  if (nullptr == this->handler_ and this->intrin_sets_["avx"].found)
    this->handler_ = this->search_(this->intrin_sets_["avx"].filename);
  if (nullptr == this->handler_)
    this->handler_ = this->search_(this->intrin_sets_["common"].filename);
}


void *LibLoader::search_(const std::string &filename) {
  std::string buffer;
  for (const auto &ld_path : this->ld_list_) {
    buffer = ld_path + filename;
    auto result_ptr = dlopen(buffer.c_str(), RTLD_NOW);
    if (nullptr != result_ptr)
      return result_ptr;
  }
  return nullptr;
}

}
