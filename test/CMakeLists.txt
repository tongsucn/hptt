# ----------------------------------------------------------------------------
# Directory setting
# ----------------------------------------------------------------------------
include_directories(test)

# ----------------------------------------------------------------------------
# Test executable target setting
# ----------------------------------------------------------------------------
add_executable(unit-test unit-test.cc)
target_link_libraries(unit-test hptc ${GTEST_LIBRARIES} pthread)

add_executable(example example.cc)
target_link_libraries(example hptc)

#add_executable(perf perf.cc)
#target_link_libraries(perf hptc)
