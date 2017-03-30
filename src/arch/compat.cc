#include <hptc/arch/compat.h>

#include <cstdint>
#include <cpuid.h>


namespace hptc {

void hptc_cpuid(const uint32_t input, uint32_t output[4]) {
  __cpuid_count(input, 0, output[0], output[1], output[2], output[3]);
}

}
