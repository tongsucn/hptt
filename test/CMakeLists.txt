# ----------------------------------------------------------------------------
# Directory setting
# ----------------------------------------------------------------------------
include_directories(inc)

# ----------------------------------------------------------------------------
# Test executable target setting
# ----------------------------------------------------------------------------
add_executable(unit-test-common unit-test-common.cc)
target_link_libraries(unit-test-common hptc ${GTEST_LIBRARIES} pthread)

add_executable(unit-test-trans unit-test-trans.cc)
target_link_libraries(unit-test-trans hptc ${GTEST_LIBRARIES} pthread)
