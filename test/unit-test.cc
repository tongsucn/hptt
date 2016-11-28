#include <gtest/gtest.h>

#include <unit-test/test_tensor_util.h>
#include <unit-test/test_tensor_wrapper.h>
// #include <unit-test/test_type_and_util.h>
#include <unit-test/operations/test_operation_base.h>
// #include <unit-test/param/test_param_trans.h>
// #include <unit-test/kernels/test_kernel_trans_avx.h>
// #include <unit-test/operations/test_operation_trans.h>

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
