# tests/CMakeLists.txt

# Enable testing
enable_testing()

# Find Google Test
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Collect all test source files
file(GLOB TEST_SRC test_*.cpp)

# Create the test executable
add_executable(runTests ${TEST_SRC})

# Include directories for tests
target_include_directories(runTests PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

# Link the test executable with the main library and Google Test
target_link_libraries(runTests PRIVATE
        quant_trading_dnn_lib
        GTest::GTest
        GTest::Main
        pthread
)

# Register the test
add_test(NAME runTests COMMAND runTests)
