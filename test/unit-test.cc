#include <gtest/gtest.h>

#include <unit-test/test_tensor_util.h>
#include <unit-test/test_tensor_wrapper.h>

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
