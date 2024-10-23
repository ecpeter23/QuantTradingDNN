cmake_minimum_required(VERSION 3.29)
project(quant_trading_dnn)

# Set C++ standard to C++26
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add subdirectories
add_subdirectory(metal-cmake)
add_subdirectory(src)
add_subdirectory(tests)
