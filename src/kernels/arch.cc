#include <hptc/kernels/arch.h>

#include <dlfcn.h>


namespace hptc {

LibLoader &LibLoader::get_loader() {
  static LibLoader loader;
  return loader;
}


void *LibLoader::dlsym(const char *symbol) {
  return ::dlsym(this->handler_, symbol);
}


LibLoader::LibLoader()
    : handler_(nullptr) {
  this->handler_ = dlopen(
      "/home/ts225456/work/thesis/hptc/build/src/libhptc_avx2.so", RTLD_NOW);
}

}
