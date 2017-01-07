# ----------------------------------------------------------------------------
# Directory setting
# ----------------------------------------------------------------------------
include_directories(inc)

# ----------------------------------------------------------------------------
# Test executable target setting
# ----------------------------------------------------------------------------
add_executable(unit-test unit-test.cc)
target_link_libraries(unit-test hptc ${GTEST_LIBRARIES} pthread)

add_executable(example example.cc)
target_link_libraries(example hptc)
