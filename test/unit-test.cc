#include <gtest/gtest.h>

// Tests on utilities
#include <hptc/unit-test/test_infra.h>

// Tests on tensor wrapper
#include <hptc/unit-test/test_tensor_util.h>
#include <hptc/unit-test/test_tensor_wrapper.h>

// Tests on micro kernels and macro kernels
#include <hptc/unit-test/kernels/test_kernel_trans_avx.h>
#include <hptc/unit-test/kernels/test_macro_kernel_trans.h>

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
