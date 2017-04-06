# ----------------------------------------------------------------------------
# Directory setting
# ----------------------------------------------------------------------------
include_directories(inc)

# ----------------------------------------------------------------------------
# Test executable target setting
# ----------------------------------------------------------------------------
add_executable(unit-test-common unit-test-common.cc)
target_link_libraries(unit-test-common hptt ${GTEST_LIBRARIES} pthread)

add_executable(unit-test-trans unit-test-trans.cc)
target_link_libraries(unit-test-trans hptt ${GTEST_LIBRARIES} pthread)
