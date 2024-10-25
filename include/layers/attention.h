#ifndef QUANT_TRADING_DNN_ATTENTION_H
#define QUANT_TRADING_DNN_ATTENTION_H

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <AppKit/AppKit.hpp>
#include <MetalKit/MetalKit.hpp>

#include "layer.h"
#include <vector>
#include <memory>

class Attention : public Layer {
public:
    Attention(MTL::Device* device, size_t input_size, size_t attention_size);
    ~Attention();
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& gradOutput) override;
    void updateWeights(double learningRate) override;
    [[nodiscard]] std::string layerType() const override { return "Attention"; }

private:
    size_t input_size_;
    size_t attention_size_;

    MTL::Device* device_;
    MTL::CommandQueue* commandQueue_;
    MTL::Library* defaultLibrary_;
    MTL::ComputePipelineState* computePipelineState_;

    // Metal Buffers for Weights
    MTL::Buffer* W_query_buffer_{};
    MTL::Buffer* W_key_buffer_{};
    MTL::Buffer* W_value_buffer_{};

    // Metal Buffers for Gradients
    MTL::Buffer* grad_W_query_buffer_{};
    MTL::Buffer* grad_W_key_buffer_{};
    MTL::Buffer* grad_W_value_buffer_{};

    // Metal Buffers for Caching
    MTL::Buffer* input_buffer_{};
    MTL::Buffer* Q_buffer_{};
    MTL::Buffer* K_buffer_{};
    MTL::Buffer* V_buffer_{};
    MTL::Buffer* scores_buffer_{};
    MTL::Buffer* weights_buffer_{};

    // Metal Compute Pipelines
    MTL::ComputePipelineState* computeQKPipelines_{};
    MTL::ComputePipelineState* computeScoresPipeline_{};
    MTL::ComputePipelineState* softmaxPipeline_{};
    MTL::ComputePipelineState* computeOutputPipeline_{};

    // Helper Methods
    void initializeWeights();
    void createMetalResources();
    void releaseMetalResources();

    // Metal kernel functions
    void computeQKV(const MTL::Buffer* inputBuffer, MTL::Buffer* QBuffer, MTL::Buffer* KBuffer, MTL::Buffer* VBuffer, size_t batch_size, size_t sequence_length);
    void computeAttentionScores(MTL::Buffer* QBuffer, MTL::Buffer* KBuffer, MTL::Buffer* scoresBuffer, size_t batch_size, size_t sequence_length);
    void applySoftmax(MTL::Buffer* scoresBuffer, MTL::Buffer* weightsBuffer, size_t batch_size, size_t sequence_length);
    void computeOutput(MTL::Buffer* weightsBuffer, MTL::Buffer* VBuffer, MTL::Buffer* outputBuffer, size_t batch_size, size_t sequence_length);

    MTL::Buffer* grad_input_buffer_{};

    // Metal Compute Pipelines for Backward Pass
    MTL::ComputePipelineState* computeGradientsOutputPipeline_{};
    MTL::ComputePipelineState* computeGradientsSoftmaxPipeline_{};
    MTL::ComputePipelineState* computeGradientsScoresPipeline_{};
    MTL::ComputePipelineState* computeGradientsQKVPipeline_{};

    // Helper Methods for Backward Pass
    void computeGradientsOutput(MTL::Buffer* gradOutputBuffer, MTL::Buffer* weightsBuffer, MTL::Buffer* VBuffer, MTL::Buffer* gradWeightsBuffer, MTL::Buffer* gradVBuffer, size_t batch_size, size_t sequence_length);
    void computeGradientsSoftmax(MTL::Buffer* gradWeightsBuffer, MTL::Buffer* weightsBuffer, MTL::Buffer* gradScoresBuffer, size_t batch_size, size_t sequence_length);
    void computeGradientsScores(MTL::Buffer* gradScoresBuffer, MTL::Buffer* QBuffer, MTL::Buffer* KBuffer, MTL::Buffer* gradQBuffer, MTL::Buffer* gradKBuffer, size_t batch_size, size_t sequence_length);
    void computeGradientsQKV(MTL::Buffer* gradQBuffer, MTL::Buffer* gradKBuffer, MTL::Buffer* gradVBuffer, MTL::Buffer* inputBuffer, MTL::Buffer* gradInputBuffer, size_t batch_size, size_t sequence_length);
};

#endif //QUANT_TRADING_DNN_ATTENTION_H
