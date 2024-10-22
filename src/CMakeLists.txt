# Find the Accelerate framework (macOS specific)
find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)

# Create a library from the source files
add_library(quant_trading_dnn_lib
        neural_network.cpp
        data_loader.cpp
        layers/layer.cpp
        layers/fully_connected.cpp
        layers/activation.cpp
        utils/math.cpp
        layers/dropout.cpp
        ../include/layers/dropout.h
        layers/batch_norm.cpp
        ../include/layers/batch_norm.h
        layers/lstm.cpp
        ../include/layers/lstm.h
        layers/conv1d.cpp
        ../include/layers/conv1d.h
        layers/attention.cpp
        ../include/layers/attention.h
        layers/residual.cpp
        ../include/layers/residual.h
        layers/dense_activation.cpp
        ../include/layers/dense_activation.h
)

# Specify include directories for the library
target_include_directories(quant_trading_dnn_lib PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

# Link the Accelerate framework to the library
target_link_libraries(quant_trading_dnn_lib PRIVATE ${ACCELERATE_FRAMEWORK})

# Create the main executable
add_executable(quant_trading_dnn main.cpp)

# Link the library to the executable
target_link_libraries(quant_trading_dnn PRIVATE quant_trading_dnn_lib)
