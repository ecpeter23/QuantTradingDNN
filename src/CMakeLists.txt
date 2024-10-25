# Find the Accelerate framework (macOS specific)
find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)

# Find the Metal and MetalPerformanceShaders libraries (macOS specific)
#find_library(METAL_LIB Metal REQUIRED)
#find_library(MPS_LIB MetalPerformanceShaders REQUIRED)

# Create a library from the source files
add_library(quant_trading_dnn_lib
        neural_network.cpp
        data_loader.cpp
        layers/layer.cpp
        layers/fully_connected.cpp
        layers/activation.cpp
        layers/batch_norm.cpp
        layers/lstm.cpp
        layers/conv1d.cpp
        layers/attention.cpp
        layers/residual.cpp
        layers/dense_activation.cpp
        layers/gru.cpp
        layers/layer_norm.cpp
        layers/trans_conv.cpp
        layers/pooling.cpp
        layers/embedding.cpp
        layers/conv2d.cpp
        layers/conv3d.cpp
        ../include/utils/types.h
        ../include/layers/util/dropout.h
        ../include/layers/batch_norm.h
        ../include/layers/lstm.h
        ../include/layers/conv1d.h
        ../include/layers/attention.h
        ../include/layers/residual.h
        ../include/layers/dense_activation.h
        ../include/layers/gru.h
        ../include/layers/layer_norm.h
        ../include/layers/trans_conv.h
        ../include/layers/pooling.h
        ../include/layers/embedding.h
        ../include/layers/conv2d.h
        ../include/layers/conv3d.h
        ../include/layers/util/flatten.h
        ../include/layers/util/reshape.h
)

# Specify include directories for the library
target_include_directories(quant_trading_dnn_lib PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

# Link the Accelerate framework to the library
target_link_libraries(quant_trading_dnn_lib PRIVATE ${ACCELERATE_FRAMEWORK} METAL_CPP)

# Create the main executable
add_executable(quant_trading_dnn main.cpp)

# Link the library to the executable
target_link_libraries(quant_trading_dnn PRIVATE quant_trading_dnn_lib)
