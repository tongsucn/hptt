#include <gtest/gtest.h>

// Tests on utilities
#include <hptt/unit-test/test_infra.h>

// Tests on tensor wrapper
#include <hptt/unit-test/test_tensor_util.h>
#include <hptt/unit-test/test_tensor_wrapper.h>


int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
