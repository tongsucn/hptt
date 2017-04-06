#include <gtest/gtest.h>

// Tests on transpose micro kernels and macro kernels
#include <hptt/unit-test/kernels/test_kernel_trans_avx.h>
#include <hptt/unit-test/kernels/test_macro_kernel_trans.h>

// Tests on transpose parameter
// #include <hptt/unit-test/param/test_parameter_trans.h>

// Tests on transpose operations
// #incdlue <hptt/unit-test/operations/test_operation_trans.h>


int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
