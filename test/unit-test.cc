#include <gtest/gtest.h>

// Tests on utilities
// #include <unit-test/test_util.h>

// Tests on tensor wrapper
// #include <unit-test/test_tensor_util.h>
#include <unit-test/test_tensor_wrapper.h>

// Tests on micro kernels and macro kernels
//#include <unit-test/kernels/test_kernel_trans.h>
//#include <unit-test/kernels/test_kernel_trans_avx.h>
//#include <unit-test/kernels/test_macro_kernel_trans.h>

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
