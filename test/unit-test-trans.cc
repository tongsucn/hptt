#include <gtest/gtest.h>

// Tests on transpose micro kernels and macro kernels
#include <hptc/unit-test/kernels/test_kernel_trans_avx.h>
#include <hptc/unit-test/kernels/test_macro_kernel_trans.h>

// Tests on transpose parameter
// #include <hptc/unit-test/param/test_parameter_trans.h>

// Tests on transpose operations
// #incdlue <hptc/unit-test/operations/test_operation_trans.h>


int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
