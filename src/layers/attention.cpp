#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <AppKit/AppKit.hpp>
#include <MetalKit/MetalKit.hpp>

#include "layers/attention.h"
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <cstring>
#include <random>

Attention::Attention(MTL::Device* device, size_t input_size, size_t attention_size)
        : input_size_(input_size), attention_size_(attention_size), device_(device) {
  if (!device_) {
    throw std::runtime_error("Metal device is null.");
  }

  commandQueue_ = device_->newCommandQueue();
  if (!commandQueue_) {
    throw std::runtime_error("Failed to create Metal command queue.");
  }

  initializeWeights();
  createMetalResources();
}

Attention::~Attention() {
  releaseMetalResources();
}

void Attention::initializeWeights() {
  // Initialize weights with small random values (CPU-side)
  std::mt19937 gen(std::random_device{}());
  std::uniform_real_distribution<> dist(-0.1, 0.1);

  size_t weight_size = input_size_ * attention_size_;
  std::vector<float> W_query_data(weight_size);
  std::vector<float> W_key_data(weight_size);
  std::vector<float> W_value_data(weight_size);

  for (size_t i = 0; i < weight_size; ++i) {
    W_query_data[i] = dist(gen);
    W_key_data[i] = dist(gen);
    W_value_data[i] = dist(gen);
  }

  // Create buffers for weights
  W_query_buffer_ = device_->newBuffer(W_query_data.data(), weight_size * sizeof(float), MTL::ResourceStorageModeShared);
  W_key_buffer_ = device_->newBuffer(W_key_data.data(), weight_size * sizeof(float), MTL::ResourceStorageModeShared);
  W_value_buffer_ = device_->newBuffer(W_value_data.data(), weight_size * sizeof(float), MTL::ResourceStorageModeShared);

  // Create buffers for gradients (initialized to zero)
  grad_W_query_buffer_ = device_->newBuffer(weight_size * sizeof(float), MTL::ResourceStorageModeShared);
  grad_W_key_buffer_ = device_->newBuffer(weight_size * sizeof(float), MTL::ResourceStorageModeShared);
  grad_W_value_buffer_ = device_->newBuffer(weight_size * sizeof(float), MTL::ResourceStorageModeShared);
}

void Attention::createMetalResources() {
  NS::Error* error = nullptr;

  // Shader code embedded as string
  const char* shaderSrc = R"(
#include <metal_stdlib>
using namespace metal;

// Forward Pass Kernels

kernel void computeQKV(
    device const float* input [[ buffer(0) ]],
    device const float* W_query [[ buffer(1) ]],
    device const float* W_key [[ buffer(2) ]],
    device const float* W_value [[ buffer(3) ]],
    device float* Q [[ buffer(4) ]],
    device float* K [[ buffer(5) ]],
    device float* V [[ buffer(6) ]],
    constant uint& input_size [[ buffer(7) ]],
    constant uint& attention_size [[ buffer(8) ]],
    constant uint& batch_size [[ buffer(9) ]],
    constant uint& sequence_length [[ buffer(10) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    uint total_elements = batch_size * sequence_length * attention_size;
    if (gid >= total_elements) return;

    uint b = gid / (sequence_length * attention_size);
    uint t = (gid / attention_size) % sequence_length;
    uint k = gid % attention_size;

    float q = 0.0f;
    float k_val = 0.0f;
    float v = 0.0f;
    for (uint i = 0; i < input_size; ++i) {
        uint idx_input = b * sequence_length * input_size + t * input_size + i;
        uint idx_weight = i * attention_size + k;
        float input_val = input[idx_input];
        q += input_val * W_query[idx_weight];
        k_val += input_val * W_key[idx_weight];
        v += input_val * W_value[idx_weight];
    }

    uint idx_QKV = b * sequence_length * attention_size + t * attention_size + k;
    Q[idx_QKV] = q;
    K[idx_QKV] = k_val;
    V[idx_QKV] = v;
}

kernel void computeAttentionScores(
    device const float* Q [[ buffer(0) ]],
    device const float* K [[ buffer(1) ]],
    device float* scores [[ buffer(2) ]],
    constant uint& attention_size [[ buffer(3) ]],
    constant uint& batch_size [[ buffer(4) ]],
    constant uint& sequence_length [[ buffer(5) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    uint total_elements = batch_size * sequence_length * sequence_length;
    if (gid >= total_elements) return;

    uint b = gid / (sequence_length * sequence_length);
    uint t = (gid / sequence_length) % sequence_length;
    uint s = gid % sequence_length;

    float score = 0.0f;
    for (uint k = 0; k < attention_size; ++k) {
        uint idx_Q = b * sequence_length * attention_size + t * attention_size + k;
        uint idx_K = b * sequence_length * attention_size + s * attention_size + k;
        score += Q[idx_Q] * K[idx_K];
    }

    // Optional scaling by sqrt(attention_size)
    score /= sqrt(float(attention_size));

    uint idx_scores = b * sequence_length * sequence_length + t * sequence_length + s;
    scores[idx_scores] = score;
}

kernel void applySoftmax(
    device const float* scores [[ buffer(0) ]],
    device float* weights [[ buffer(1) ]],
    constant uint& batch_size [[ buffer(2) ]],
    constant uint& sequence_length [[ buffer(3) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    uint total_elements = batch_size * sequence_length;
    if (gid >= total_elements) return;

    uint b = gid / sequence_length;
    uint t = gid % sequence_length;

    // Compute max for numerical stability
    float max_score = -INFINITY;
    uint base_idx = b * sequence_length * sequence_length + t * sequence_length;
    for (uint s = 0; s < sequence_length; ++s) {
        float score = scores[base_idx + s];
        if (score > max_score) {
            max_score = score;
        }
    }

    // Compute sum of exp(scores)
    float sum_exp = 0.0f;
    for (uint s = 0; s < sequence_length; ++s) {
        float exp_score = exp(scores[base_idx + s] - max_score);
        sum_exp += exp_score;
        weights[base_idx + s] = exp_score; // Temporarily store exp(scores)
    }

    // Normalize to get softmax probabilities
    for (uint s = 0; s < sequence_length; ++s) {
        weights[base_idx + s] /= sum_exp;
    }
}

kernel void computeOutput(
    device const float* weights [[ buffer(0) ]],
    device const float* V [[ buffer(1) ]],
    device float* output [[ buffer(2) ]],
    constant uint& attention_size [[ buffer(3) ]],
    constant uint& batch_size [[ buffer(4) ]],
    constant uint& sequence_length [[ buffer(5) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    uint total_elements = batch_size * sequence_length * attention_size;
    if (gid >= total_elements) return;

    uint b = gid / (sequence_length * attention_size);
    uint t = (gid / attention_size) % sequence_length;
    uint k = gid % attention_size;

    float out = 0.0f;
    uint base_weight_idx = b * sequence_length * sequence_length + t * sequence_length;
    for (uint s = 0; s < sequence_length; ++s) {
        uint idx_weight = base_weight_idx + s;
        uint idx_V = b * sequence_length * attention_size + s * attention_size + k;
        out += weights[idx_weight] * V[idx_V];
    }

    uint idx_output = b * sequence_length * attention_size + t * attention_size + k;
    output[idx_output] = out;
}

// Backward Pass Kernels

kernel void computeGradientsOutput(
    device const float* gradOutput [[ buffer(0) ]],
    device const float* weights [[ buffer(1) ]],
    device const float* V [[ buffer(2) ]],
    device float* gradWeights [[ buffer(3) ]],
    device float* gradV [[ buffer(4) ]],
    constant uint& attention_size [[ buffer(5) ]],
    constant uint& batch_size [[ buffer(6) ]],
    constant uint& sequence_length [[ buffer(7) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    uint total_elements = batch_size * sequence_length * attention_size;
    if (gid >= total_elements) return;

    uint b = gid / (sequence_length * attention_size);
    uint t = (gid / attention_size) % sequence_length;
    uint k = gid % attention_size;

    uint idx_gradOutput = b * sequence_length * attention_size + t * attention_size + k;
    float grad_out = gradOutput[idx_gradOutput];

    // Compute gradWeights and gradV
    uint base_weight_idx = b * sequence_length * sequence_length + t * sequence_length;
    for (uint s = 0; s < sequence_length; ++s) {
        uint idx_weight = base_weight_idx + s;
        uint idx_V = b * sequence_length * attention_size + s * attention_size + k;

        // Accumulate gradWeights
        atomic_fetch_add_explicit(&(gradWeights[idx_weight]), grad_out * V[idx_V], memory_order_relaxed);

        // Accumulate gradV
        atomic_fetch_add_explicit(&(gradV[idx_V]), grad_out * weights[idx_weight], memory_order_relaxed);
    }
}

kernel void computeGradientsSoftmax(
    device const float* gradWeights [[ buffer(0) ]],
    device const float* weights [[ buffer(1) ]],
    device float* gradScores [[ buffer(2) ]],
    constant uint& batch_size [[ buffer(3) ]],
    constant uint& sequence_length [[ buffer(4) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    uint total_elements = batch_size * sequence_length * sequence_length;
    if (gid >= total_elements) return;

    uint b = gid / (sequence_length * sequence_length);
    uint t = (gid / sequence_length) % sequence_length;
    uint s = gid % sequence_length;

    uint idx = b * sequence_length * sequence_length + t * sequence_length + s;
    float weight = weights[idx];
    float grad_weight = gradWeights[idx];

    // Compute sum over the sequence_length
    float sum = 0.0f;
    uint base_idx = b * sequence_length * sequence_length + t * sequence_length;
    for (uint i = 0; i < sequence_length; ++i) {
        sum += gradWeights[base_idx + i] * weights[base_idx + i];
    }

    gradScores[idx] = (grad_weight - sum) * weight;
}

kernel void computeGradientsScores(
    device const float* gradScores [[ buffer(0) ]],
    device const float* Q [[ buffer(1) ]],
    device const float* K [[ buffer(2) ]],
    device float* gradQ [[ buffer(3) ]],
    device float* gradK [[ buffer(4) ]],
    constant uint& attention_size [[ buffer(5) ]],
    constant uint& batch_size [[ buffer(6) ]],
    constant uint& sequence_length [[ buffer(7) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    uint total_elements = batch_size * sequence_length * attention_size;
    if (gid >= total_elements) return;

    uint b = gid / (sequence_length * attention_size);
    uint t = (gid / attention_size) % sequence_length;
    uint k = gid % attention_size;

    // Initialize gradients
    float grad_q = 0.0f;
    float grad_k = 0.0f;

    uint base_scores_idx = b * sequence_length * sequence_length + t * sequence_length;
    for (uint s = 0; s < sequence_length; ++s) {
        uint idx_scores = base_scores_idx + s;
        float grad_score = gradScores[idx_scores];
        uint idx_K = b * sequence_length * attention_size + s * attention_size + k;
        grad_q += grad_score * K[idx_K];
    }

    for (uint s = 0; s < sequence_length; ++s) {
        uint idx_scores = b * sequence_length * sequence_length + s * sequence_length + t;
        float grad_score = gradScores[idx_scores];
        uint idx_Q = b * sequence_length * attention_size + s * attention_size + k;
        grad_k += grad_score * Q[idx_Q];
    }

    grad_q /= sqrt(float(attention_size));
    grad_k /= sqrt(float(attention_size));

    uint idx_gradQK = b * sequence_length * attention_size + t * attention_size + k;
    gradQ[idx_gradQK] = grad_q;
    gradK[idx_gradQK] = grad_k;
}

kernel void computeGradientsQKV(
    device const float* gradQ [[ buffer(0) ]],
    device const float* gradK [[ buffer(1) ]],
    device const float* gradV [[ buffer(2) ]],
    device const float* input [[ buffer(3) ]],
    device float* gradInput [[ buffer(4) ]],
    device float* grad_W_query [[ buffer(5) ]],
    device float* grad_W_key [[ buffer(6) ]],
    device float* grad_W_value [[ buffer(7) ]],
    constant uint& input_size [[ buffer(8) ]],
    constant uint& attention_size [[ buffer(9) ]],
    constant uint& batch_size [[ buffer(10) ]],
    constant uint& sequence_length [[ buffer(11) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    uint total_elements = batch_size * sequence_length * input_size;
    if (gid >= total_elements) return;

    uint b = gid / (sequence_length * input_size);
    uint t = (gid / input_size) % sequence_length;
    uint i = gid % input_size;

    float grad_input = 0.0f;

    // Compute grad_input and grad_W_query, grad_W_key, grad_W_value
    for (uint k = 0; k < attention_size; ++k) {
        uint idx_input = b * sequence_length * input_size + t * input_size + i;
        uint idx_gradQKV = b * sequence_length * attention_size + t * attention_size + k;

        float input_val = input[idx_input];
        float grad_q = gradQ[idx_gradQKV];
        float grad_k = gradK[idx_gradQKV];
        float grad_v = gradV[idx_gradQKV];

        // Update gradients w.r.t weights
        uint idx_weight = i * attention_size + k;
        atomic_fetch_add_explicit(&(grad_W_query[idx_weight]), input_val * grad_q, memory_order_relaxed);
        atomic_fetch_add_explicit(&(grad_W_key[idx_weight]), input_val * grad_k, memory_order_relaxed);
        atomic_fetch_add_explicit(&(grad_W_value[idx_weight]), input_val * grad_v, memory_order_relaxed);

        // Accumulate grad_input
        grad_input += grad_q * W_query[idx_weight] + grad_k * W_key[idx_weight] + grad_v * W_value[idx_weight];
    }

    uint idx_gradInput = b * sequence_length * input_size + t * input_size + i;
    gradInput[idx_gradInput] = grad_input;
}
)";
  defaultLibrary_ = device_->newLibrary(NS::String::string(shaderSrc, NS::StringEncoding::UTF8StringEncoding), nullptr, &error);
  if (!defaultLibrary_) {
    std::cerr << "Failed to compile shader library: " << error->localizedDescription()->utf8String() << std::endl;
    return;
  }

  // Forward Pipeline States
  // ComputeQKV
  MTL::Function* computeQKVFunction = defaultLibrary_->newFunction(NS::String::string("computeQKV", NS::UTF8StringEncoding));
  if (!computeQKVFunction) {
    std::cerr << "Failed to find computeQKV function in the Metal library." << std::endl;
    return;
  }
  computePipelineState_ = device_->newComputePipelineState(computeQKVFunction, &error);
  if (!computePipelineState_) {
    std::cerr << "Failed to create computeQKV pipeline state: " << error->localizedDescription()->utf8String() << std::endl;
    return;
  }

  // ComputeAttentionScores
  MTL::Function* computeScoresFunction = defaultLibrary_->newFunction(NS::String::string("computeAttentionScores", NS::UTF8StringEncoding));
  if (!computeScoresFunction) {
    std::cerr << "Failed to find computeAttentionScores function in the Metal library." << std::endl;
    return;
  }
  computeScoresPipeline_ = device_->newComputePipelineState(computeScoresFunction, &error);
  if (!computeScoresPipeline_) {
    std::cerr << "Failed to create computeScores pipeline state: " << error->localizedDescription()->utf8String() << std::endl;
    return;
  }

  // ApplySoftmax
  MTL::Function* softmaxFunction = defaultLibrary_->newFunction(NS::String::string("applySoftmax", NS::UTF8StringEncoding));
  if (!softmaxFunction) {
    std::cerr << "Failed to find applySoftmax function in the Metal library." << std::endl;
    return;
  }
  softmaxPipeline_ = device_->newComputePipelineState(softmaxFunction, &error);
  if (!softmaxPipeline_) {
    std::cerr << "Failed to create softmax pipeline state: " << error->localizedDescription()->utf8String() << std::endl;
    return;
  }

  // ComputeOutput
  MTL::Function* computeOutputFunction = defaultLibrary_->newFunction(NS::String::string("computeOutput", NS::UTF8StringEncoding));
  if (!computeOutputFunction) {
    std::cerr << "Failed to find computeOutput function in the Metal library." << std::endl;
    return;
  }
  computeOutputPipeline_ = device_->newComputePipelineState(computeOutputFunction, &error);
  if (!computeOutputPipeline_) {
    std::cerr << "Failed to create computeOutput pipeline state: " << error->localizedDescription()->utf8String() << std::endl;
    return;
  }

  // Backward Pipeline States
  // ComputeGradientsOutput
  MTL::Function* computeGradientsOutputFunction = defaultLibrary_->newFunction(NS::String::string("computeGradientsOutput", NS::UTF8StringEncoding));
  if (!computeGradientsOutputFunction) {
    std::cerr << "Failed to find computeGradientsOutput function in the Metal library." << std::endl;
    return;
  }
  computeGradientsOutputPipeline_ = device_->newComputePipelineState(computeGradientsOutputFunction, &error);
  if (!computeGradientsOutputPipeline_) {
    std::cerr << "Failed to create computeGradientsOutput pipeline state: " << error->localizedDescription()->utf8String() << std::endl;
    return;
  }

  // ComputeGradientsSoftmax
  MTL::Function* computeGradientsSoftmaxFunction = defaultLibrary_->newFunction(NS::String::string("computeGradientsSoftmax", NS::UTF8StringEncoding));
  if (!computeGradientsSoftmaxFunction) {
    std::cerr << "Failed to find computeGradientsSoftmax function in the Metal library." << std::endl;
    return;
  }
  computeGradientsSoftmaxPipeline_ = device_->newComputePipelineState(computeGradientsSoftmaxFunction, &error);
  if (!computeGradientsSoftmaxPipeline_) {
    std::cerr << "Failed to create computeGradientsSoftmax pipeline state: " << error->localizedDescription()->utf8String() << std::endl;
    return;
  }

  // ComputeGradientsScores
  MTL::Function* computeGradientsScoresFunction = defaultLibrary_->newFunction(NS::String::string("computeGradientsScores", NS::UTF8StringEncoding));
  if (!computeGradientsScoresFunction) {
    std::cerr << "Failed to find computeGradientsScores function in the Metal library." << std::endl;
    return;
  }
  computeGradientsScoresPipeline_ = device_->newComputePipelineState(computeGradientsScoresFunction, &error);
  if (!computeGradientsScoresPipeline_) {
    std::cerr << "Failed to create computeGradientsScores pipeline state: " << error->localizedDescription()->utf8String() << std::endl;
    return;
  }

  // ComputeGradientsQKV
  MTL::Function* computeGradientsQKVFunction = defaultLibrary_->newFunction(NS::String::string("computeGradientsQKV", NS::UTF8StringEncoding));
  if (!computeGradientsQKVFunction) {
    std::cerr << "Failed to find computeGradientsQKV function in the Metal library." << std::endl;
    return;
  }
  computeGradientsQKVPipeline_ = device_->newComputePipelineState(computeGradientsQKVFunction, &error);
  if (!computeGradientsQKVPipeline_) {
    std::cerr << "Failed to create computeGradientsQKV pipeline state: " << error->localizedDescription()->utf8String() << std::endl;
    return;
  }
}


void Attention::releaseMetalResources() {
  // Release Metal resources
  if (W_query_buffer_) W_query_buffer_->release();
  if (W_key_buffer_) W_key_buffer_->release();
  if (W_value_buffer_) W_value_buffer_->release();

  if (grad_W_query_buffer_) grad_W_query_buffer_->release();
  if (grad_W_key_buffer_) grad_W_key_buffer_->release();
  if (grad_W_value_buffer_) grad_W_value_buffer_->release();

  if (commandQueue_) commandQueue_->release();
  if (defaultLibrary_) defaultLibrary_->release();
  if (computePipelineState_) computePipelineState_->release();

  if (computeScoresPipeline_) computeScoresPipeline_->release();
  if (softmaxPipeline_) softmaxPipeline_->release();
  if (computeOutputPipeline_) computeOutputPipeline_->release();

  if (input_buffer_) input_buffer_->release();
  if (Q_buffer_) Q_buffer_->release();
  if (K_buffer_) K_buffer_->release();

  if (V_buffer_) V_buffer_->release();
  if (scores_buffer_) scores_buffer_->release();
  if (weights_buffer_) weights_buffer_->release();

  if (computeQKPipelines_) computeQKPipelines_->release();

  if (computeGradientsOutputPipeline_) computeGradientsOutputPipeline_->release();
  if (computeGradientsSoftmaxPipeline_) computeGradientsSoftmaxPipeline_->release();
  if (computeGradientsScoresPipeline_) computeGradientsScoresPipeline_->release();
  if (computeGradientsQKVPipeline_) computeGradientsQKVPipeline_->release();
}

Layer::Tensor Attention::forward(const Tensor& input) {
  // Extract input data
  auto input_ptr = std::get_if<std::vector<std::vector<std::vector<double>>>>(&input);
  if (!input_ptr) {
    throw std::invalid_argument("Attention layer expects input to be a 3D tensor (batch_size x sequence_length x input_size).");
  }
  const auto& input_tensor = *input_ptr;

  size_t batch_size = input_tensor.size();
  size_t sequence_length = input_tensor[0].size();

  // Flatten input data for GPU
  size_t input_data_size = batch_size * sequence_length * input_size_;
  std::vector<float> input_data(input_data_size);

  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t t = 0; t < sequence_length; ++t) {
      for (size_t i = 0; i < input_size_; ++i) {
        input_data[b * sequence_length * input_size_ + t * input_size_ + i] = static_cast<float>(input_tensor[b][t][i]);
      }
    }
  }

  // Create input buffer
  MTL::Buffer* inputBuffer = device_->newBuffer(input_data.data(), input_data_size * sizeof(float), MTL::ResourceStorageModeShared);

  // Create buffers for Q, K, V
  size_t QKV_size = batch_size * sequence_length * attention_size_ * sizeof(float);
  MTL::Buffer* QBuffer = device_->newBuffer(QKV_size, MTL::ResourceStorageModeShared);
  MTL::Buffer* KBuffer = device_->newBuffer(QKV_size, MTL::ResourceStorageModeShared);
  MTL::Buffer* VBuffer = device_->newBuffer(QKV_size, MTL::ResourceStorageModeShared);

  // Compute Q, K, V
  Attention::computeQKV(inputBuffer, QBuffer, KBuffer, VBuffer, batch_size, sequence_length);

  // Compute attention scores
  size_t scores_size = batch_size * sequence_length * sequence_length * sizeof(float);
  MTL::Buffer* scoresBuffer = device_->newBuffer(scores_size, MTL::ResourceStorageModeShared);

  Attention::computeAttentionScores(QBuffer, KBuffer, scoresBuffer, batch_size, sequence_length);

  // Apply softmax to get attention weights
  MTL::Buffer* weightsBuffer = device_->newBuffer(scores_size, MTL::ResourceStorageModeShared);

  Attention::applySoftmax(scoresBuffer, weightsBuffer, batch_size, sequence_length);

  // Compute the output
  size_t output_size = batch_size * sequence_length * attention_size_ * sizeof(float);
  MTL::Buffer* outputBuffer = device_->newBuffer(output_size, MTL::ResourceStorageModeShared);

  Attention::computeOutput(weightsBuffer, VBuffer, outputBuffer, batch_size, sequence_length);

  // Retrieve output data from GPU
  std::vector<float> output_data(output_size / sizeof(float));
  memcpy(output_data.data(), outputBuffer->contents(), output_size);

  // Convert output_data back to the expected Tensor format
  std::vector<std::vector<std::vector<double>>> output(batch_size, std::vector<std::vector<double>>(sequence_length, std::vector<double>(attention_size_)));

  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t t = 0; t < sequence_length; ++t) {
      for (size_t k = 0; k < attention_size_; ++k) {
        output[b][t][k] = static_cast<double>(output_data[b * sequence_length * attention_size_ + t * attention_size_ + k]);
      }
    }
  }

  // Release buffers
  inputBuffer->release();
  QBuffer->release();
  KBuffer->release();
  VBuffer->release();
  scoresBuffer->release();
  weightsBuffer->release();
  outputBuffer->release();

  // Return output as Tensor
  return output;
}

void Attention::computeQKV(const MTL::Buffer* inputBuffer, MTL::Buffer* QBuffer, MTL::Buffer* KBuffer, MTL::Buffer* VBuffer, size_t batch_size, size_t sequence_length) {
  // Create command buffer and encoder
  MTL::CommandBuffer* commandBuffer = commandQueue_->commandBuffer();
  MTL::ComputeCommandEncoder* computeEncoder = commandBuffer->computeCommandEncoder();

  // Set pipeline state
  computeEncoder->setComputePipelineState(computePipelineState_);

  // Set buffers
  computeEncoder->setBuffer(const_cast<MTL::Buffer*>(inputBuffer), 0, 0);
  computeEncoder->setBuffer(W_query_buffer_, 0, 1);
  computeEncoder->setBuffer(W_key_buffer_, 0, 2);
  computeEncoder->setBuffer(W_value_buffer_, 0, 3);
  computeEncoder->setBuffer(QBuffer, 0, 4);
  computeEncoder->setBuffer(KBuffer, 0, 5);
  computeEncoder->setBuffer(VBuffer, 0, 6);

  // Set constants
  uint input_size = static_cast<uint>(input_size_);
  uint attention_size = static_cast<uint>(attention_size_);
  uint batch_size_uint = static_cast<uint>(batch_size);
  uint sequence_length_uint = static_cast<uint>(sequence_length);
  computeEncoder->setBytes(&input_size, sizeof(uint), 7);
  computeEncoder->setBytes(&attention_size, sizeof(uint), 8);
  computeEncoder->setBytes(&batch_size_uint, sizeof(uint), 9);
  computeEncoder->setBytes(&sequence_length_uint, sizeof(uint), 10);

  // Dispatch threads
  size_t totalThreads = batch_size * sequence_length * attention_size_;
  MTL::Size gridSize(totalThreads, 1, 1);
  MTL::Size threadGroupSize = MTL::Size::Make(computePipelineState_->maxTotalThreadsPerThreadgroup(), 1, 1);

  computeEncoder->dispatchThreads(gridSize, threadGroupSize);
  computeEncoder->endEncoding();
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();
}

void Attention::computeAttentionScores(MTL::Buffer* QBuffer, MTL::Buffer* KBuffer,
                                       MTL::Buffer* scoresBuffer, size_t batch_size,
                                       size_t sequence_length) {
  MTL::CommandBuffer* commandBuffer = commandQueue_->commandBuffer();
  MTL::ComputeCommandEncoder* computeEncoder = commandBuffer->computeCommandEncoder();

  computeEncoder->setComputePipelineState(computeScoresPipeline_);
  computeEncoder->setBuffer(QBuffer, 0, 0);
  computeEncoder->setBuffer(KBuffer, 0, 1);
  computeEncoder->setBuffer(scoresBuffer, 0, 2);

  // Set constants
  uint attention_size = static_cast<uint>(attention_size_);
  uint batch_size_uint = static_cast<uint>(batch_size);
  uint sequence_length_uint = static_cast<uint>(sequence_length);
  computeEncoder->setBytes(&attention_size, sizeof(uint), 3);
  computeEncoder->setBytes(&batch_size_uint, sizeof(uint), 4);
  computeEncoder->setBytes(&sequence_length_uint, sizeof(uint), 5);

  size_t totalThreads = batch_size * sequence_length * sequence_length;
  MTL::Size gridSize(totalThreads, 1, 1);
  MTL::Size threadGroupSize = MTL::Size::Make(computeScoresPipeline_->maxTotalThreadsPerThreadgroup(), 1, 1);

  computeEncoder->dispatchThreads(gridSize, threadGroupSize);
  computeEncoder->endEncoding();
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();
}

void Attention::applySoftmax(MTL::Buffer* scoresBuffer, MTL::Buffer* weightsBuffer,
                             size_t batch_size, size_t sequence_length) {
  MTL::CommandBuffer* commandBuffer = commandQueue_->commandBuffer();
  MTL::ComputeCommandEncoder* computeEncoder = commandBuffer->computeCommandEncoder();

  computeEncoder->setComputePipelineState(softmaxPipeline_);
  computeEncoder->setBuffer(scoresBuffer, 0, 0);
  computeEncoder->setBuffer(weightsBuffer, 0, 1);

  // Set constants
  uint batch_size_uint = static_cast<uint>(batch_size);
  uint sequence_length_uint = static_cast<uint>(sequence_length);
  computeEncoder->setBytes(&batch_size_uint, sizeof(uint), 2);
  computeEncoder->setBytes(&sequence_length_uint, sizeof(uint), 3);

  size_t totalThreads = batch_size * sequence_length;
  MTL::Size gridSize(totalThreads, 1, 1);
  MTL::Size threadGroupSize = MTL::Size::Make(softmaxPipeline_->maxTotalThreadsPerThreadgroup(), 1, 1);

  computeEncoder->dispatchThreads(gridSize, threadGroupSize);
  computeEncoder->endEncoding();
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();
}

void Attention::computeOutput(MTL::Buffer* weightsBuffer, MTL::Buffer* VBuffer,
                              MTL::Buffer* outputBuffer, size_t batch_size,
                              size_t sequence_length) {
  MTL::CommandBuffer* commandBuffer = commandQueue_->commandBuffer();
  MTL::ComputeCommandEncoder* computeEncoder = commandBuffer->computeCommandEncoder();

  computeEncoder->setComputePipelineState(computeOutputPipeline_);
  computeEncoder->setBuffer(weightsBuffer, 0, 0);
  computeEncoder->setBuffer(VBuffer, 0, 1);
  computeEncoder->setBuffer(outputBuffer, 0, 2);

  // Set constants
  uint attention_size = static_cast<uint>(attention_size_);
  uint batch_size_uint = static_cast<uint>(batch_size);
  uint sequence_length_uint = static_cast<uint>(sequence_length);
  computeEncoder->setBytes(&attention_size, sizeof(uint), 3);
  computeEncoder->setBytes(&batch_size_uint, sizeof(uint), 4);
  computeEncoder->setBytes(&sequence_length_uint, sizeof(uint), 5);

  size_t totalThreads = batch_size * sequence_length * attention_size_;
  MTL::Size gridSize(totalThreads, 1, 1);
  MTL::Size threadGroupSize = MTL::Size::Make(computeOutputPipeline_->maxTotalThreadsPerThreadgroup(), 1, 1);

  computeEncoder->dispatchThreads(gridSize, threadGroupSize);
  computeEncoder->endEncoding();
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();
}

void Attention::updateWeights(double learningRate) {
  // Update weights on the GPU or transfer gradients to CPU to update

  // For simplicity, we'll assume gradients are computed and stored on the CPU

  size_t weight_size = input_size_ * attention_size_;

  // Map gradient buffers to CPU memory
  auto* grad_W_query = static_cast<float*>(grad_W_query_buffer_->contents());
  auto* grad_W_key = static_cast<float*>(grad_W_key_buffer_->contents());
  auto* grad_W_value = static_cast<float*>(grad_W_value_buffer_->contents());

  // Map weight buffers
  auto* W_query = static_cast<float*>(W_query_buffer_->contents());
  auto* W_key = static_cast<float*>(W_key_buffer_->contents());
  auto* W_value = static_cast<float*>(W_value_buffer_->contents());

  // Update weights
  for (size_t i = 0; i < weight_size; ++i) {
    W_query[i] -= learningRate * grad_W_query[i];
    W_key[i] -= learningRate * grad_W_key[i];
    W_value[i] -= learningRate * grad_W_value[i];

    // Reset gradients
    grad_W_query[i] = 0.0f;
    grad_W_key[i] = 0.0f;
    grad_W_value[i] = 0.0f;
  }
}

Layer::Tensor Attention::backward(const Tensor& gradOutput) {
  // Extract gradOutput data
  auto gradOutput_ptr = std::get_if<std::vector<std::vector<std::vector<double>>>>(&gradOutput);
  if (!gradOutput_ptr) {
    throw std::invalid_argument("Attention layer expects gradOutput to be a 3D tensor (batch_size x sequence_length x attention_size).");
  }
  const auto& gradOutput_tensor = *gradOutput_ptr;

  size_t batch_size = gradOutput_tensor.size();
  size_t sequence_length = gradOutput_tensor[0].size();

  // Flatten gradOutput data for GPU
  size_t gradOutput_data_size = batch_size * sequence_length * attention_size_;
  std::vector<float> gradOutput_data(gradOutput_data_size);

  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t t = 0; t < sequence_length; ++t) {
      for (size_t k = 0; k < attention_size_; ++k) {
        gradOutput_data[b * sequence_length * attention_size_ + t * attention_size_ + k] = static_cast<float>(gradOutput_tensor[b][t][k]);
      }
    }
  }

  // Create gradOutput buffer
  MTL::Buffer* gradOutputBuffer = device_->newBuffer(gradOutput_data.data(), gradOutput_data_size * sizeof(float), MTL::ResourceStorageModeShared);

  // Create buffers for gradients
  size_t QKV_size = batch_size * sequence_length * attention_size_ * sizeof(float);
  MTL::Buffer* gradVBuffer = device_->newBuffer(QKV_size, MTL::ResourceStorageModeShared);
  MTL::Buffer* gradWeightsBuffer = device_->newBuffer(batch_size * sequence_length * sequence_length * sizeof(float), MTL::ResourceStorageModeShared);
  MTL::Buffer* gradScoresBuffer = device_->newBuffer(batch_size * sequence_length * sequence_length * sizeof(float), MTL::ResourceStorageModeShared);
  MTL::Buffer* gradQBuffer = device_->newBuffer(QKV_size, MTL::ResourceStorageModeShared);
  MTL::Buffer* gradKBuffer = device_->newBuffer(QKV_size, MTL::ResourceStorageModeShared);
  MTL::Buffer* gradInputBuffer = device_->newBuffer(batch_size * sequence_length * input_size_ * sizeof(float), MTL::ResourceStorageModeShared);

  // Compute gradients w.r.t output
  computeGradientsOutput(gradOutputBuffer, weights_buffer_, V_buffer_, gradWeightsBuffer, gradVBuffer, batch_size, sequence_length);

  // Compute gradients w.r.t softmax
  computeGradientsSoftmax(gradWeightsBuffer, weights_buffer_, gradScoresBuffer, batch_size, sequence_length);

  // Compute gradients w.r.t attention scores
  computeGradientsScores(gradScoresBuffer, Q_buffer_, K_buffer_, gradQBuffer, gradKBuffer, batch_size, sequence_length);

  // Compute gradients w.r.t Q, K, V and input
  computeGradientsQKV(gradQBuffer, gradKBuffer, gradVBuffer, input_buffer_, gradInputBuffer, batch_size, sequence_length);

  // Update gradients for weights (W_query, W_key, W_value)
  // (Assuming weights are updated on CPU for simplicity)

  // Map gradient buffers to CPU memory
  auto* grad_W_query = static_cast<float*>(grad_W_query_buffer_->contents());
  auto* grad_W_key = static_cast<float*>(grad_W_key_buffer_->contents());
  auto* grad_W_value = static_cast<float*>(grad_W_value_buffer_->contents());

  // Map input buffer
  auto* input_data = static_cast<float*>(input_buffer_->contents());
  auto* gradQ_data = static_cast<float*>(gradQBuffer->contents());
  auto* gradK_data = static_cast<float*>(gradKBuffer->contents());
  auto* gradV_data = static_cast<float*>(gradVBuffer->contents());

  // Compute gradients w.r.t weights
  // For simplicity, perform this on CPU
  size_t weight_size = input_size_ * attention_size_;
  memset(grad_W_query, 0, weight_size * sizeof(float));
  memset(grad_W_key, 0, weight_size * sizeof(float));
  memset(grad_W_value, 0, weight_size * sizeof(float));

  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t t = 0; t < sequence_length; ++t) {
      for (size_t i = 0; i < input_size_; ++i) {
        for (size_t k = 0; k < attention_size_; ++k) {
          size_t idx_input = b * sequence_length * input_size_ + t * input_size_ + i;
          size_t idx_gradQKV = b * sequence_length * attention_size_ + t * attention_size_ + k;
          grad_W_query[i * attention_size_ + k] += input_data[idx_input] * gradQ_data[idx_gradQKV];
          grad_W_key[i * attention_size_ + k] += input_data[idx_input] * gradK_data[idx_gradQKV];
          grad_W_value[i * attention_size_ + k] += input_data[idx_input] * gradV_data[idx_gradQKV];
        }
      }
    }
  }

  // Retrieve gradInput data from GPU
  size_t gradInput_size = batch_size * sequence_length * input_size_;
  std::vector<float> gradInput_data(gradInput_size);
  memcpy(gradInput_data.data(), gradInputBuffer->contents(), gradInput_size * sizeof(float));

  // Convert gradInput_data back to the expected Tensor format
  std::vector<std::vector<std::vector<double>>> gradInput(batch_size, std::vector<std::vector<double>>(sequence_length, std::vector<double>(input_size_)));

  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t t = 0; t < sequence_length; ++t) {
      for (size_t i = 0; i < input_size_; ++i) {
        gradInput[b][t][i] = static_cast<double>(gradInput_data[b * sequence_length * input_size_ + t * input_size_ + i]);
      }
    }
  }

  // Release buffers
  gradOutputBuffer->release();
  gradVBuffer->release();
  gradWeightsBuffer->release();
  gradScoresBuffer->release();
  gradQBuffer->release();
  gradKBuffer->release();
  gradInputBuffer->release();

  // Return gradInput as Tensor
  return gradInput;
}

void Attention::computeGradientsOutput(MTL::Buffer* gradOutputBuffer, MTL::Buffer* weightsBuffer, MTL::Buffer* VBuffer, MTL::Buffer* gradWeightsBuffer, MTL::Buffer* gradVBuffer, size_t batch_size, size_t sequence_length) {
  // Create command buffer and encoder
  MTL::CommandBuffer* commandBuffer = commandQueue_->commandBuffer();
  MTL::ComputeCommandEncoder* computeEncoder = commandBuffer->computeCommandEncoder();

  // Set pipeline state
  computeEncoder->setComputePipelineState(computeGradientsOutputPipeline_);

  // Set buffers
  computeEncoder->setBuffer(gradOutputBuffer, 0, 0);
  computeEncoder->setBuffer(weightsBuffer, 0, 1);
  computeEncoder->setBuffer(VBuffer, 0, 2);
  computeEncoder->setBuffer(gradWeightsBuffer, 0, 3);
  computeEncoder->setBuffer(gradVBuffer, 0, 4);

  // Set constants
  uint attention_size = static_cast<uint>(attention_size_);
  uint batch_size_uint = static_cast<uint>(batch_size);
  uint sequence_length_uint = static_cast<uint>(sequence_length);
  computeEncoder->setBytes(&attention_size, sizeof(uint), 5);
  computeEncoder->setBytes(&batch_size_uint, sizeof(uint), 6);
  computeEncoder->setBytes(&sequence_length_uint, sizeof(uint), 7);

  // Dispatch threads
  size_t totalThreads = batch_size * sequence_length * attention_size_;
  MTL::Size gridSize(totalThreads, 1, 1);
  MTL::Size threadGroupSize = MTL::Size::Make(computeGradientsOutputPipeline_->maxTotalThreadsPerThreadgroup(), 1, 1);

  computeEncoder->dispatchThreads(gridSize, threadGroupSize);
  computeEncoder->endEncoding();
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();
}

void Attention::computeGradientsSoftmax(MTL::Buffer* gradWeightsBuffer, MTL::Buffer* weightsBuffer, MTL::Buffer* gradScoresBuffer, size_t batch_size, size_t sequence_length) {
  // Create command buffer and encoder
  MTL::CommandBuffer* commandBuffer = commandQueue_->commandBuffer();
  MTL::ComputeCommandEncoder* computeEncoder = commandBuffer->computeCommandEncoder();

  // Set pipeline state
  computeEncoder->setComputePipelineState(computeGradientsSoftmaxPipeline_);

  // Set buffers
  computeEncoder->setBuffer(gradWeightsBuffer, 0, 0);
  computeEncoder->setBuffer(weightsBuffer, 0, 1);
  computeEncoder->setBuffer(gradScoresBuffer, 0, 2);

  // Set constants
  uint batch_size_uint = static_cast<uint>(batch_size);
  uint sequence_length_uint = static_cast<uint>(sequence_length);
  computeEncoder->setBytes(&batch_size_uint, sizeof(uint), 3);
  computeEncoder->setBytes(&sequence_length_uint, sizeof(uint), 4);

  // Dispatch threads
  size_t totalThreads = batch_size * sequence_length * sequence_length;
  MTL::Size gridSize(totalThreads, 1, 1);
  MTL::Size threadGroupSize = MTL::Size::Make(computeGradientsSoftmaxPipeline_->maxTotalThreadsPerThreadgroup(), 1, 1);

  computeEncoder->dispatchThreads(gridSize, threadGroupSize);
  computeEncoder->endEncoding();
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();
}

void Attention::computeGradientsScores(MTL::Buffer* gradScoresBuffer, MTL::Buffer* QBuffer, MTL::Buffer* KBuffer, MTL::Buffer* gradQBuffer, MTL::Buffer* gradKBuffer, size_t batch_size, size_t sequence_length) {
  // Create command buffer and encoder
  MTL::CommandBuffer* commandBuffer = commandQueue_->commandBuffer();
  MTL::ComputeCommandEncoder* computeEncoder = commandBuffer->computeCommandEncoder();

  // Set pipeline state
  computeEncoder->setComputePipelineState(computeGradientsScoresPipeline_);

  // Set buffers
  computeEncoder->setBuffer(gradScoresBuffer, 0, 0);
  computeEncoder->setBuffer(QBuffer, 0, 1);
  computeEncoder->setBuffer(KBuffer, 0, 2);
  computeEncoder->setBuffer(gradQBuffer, 0, 3);
  computeEncoder->setBuffer(gradKBuffer, 0, 4);

  // Set constants
  uint attention_size = static_cast<uint>(attention_size_);
  uint batch_size_uint = static_cast<uint>(batch_size);
  uint sequence_length_uint = static_cast<uint>(sequence_length);
  computeEncoder->setBytes(&attention_size, sizeof(uint), 5);
  computeEncoder->setBytes(&batch_size_uint, sizeof(uint), 6);
  computeEncoder->setBytes(&sequence_length_uint, sizeof(uint), 7);

  // Dispatch threads
  size_t totalThreads = batch_size * sequence_length * attention_size_;
  MTL::Size gridSize(totalThreads, 1, 1);
  MTL::Size threadGroupSize = MTL::Size::Make(computeGradientsScoresPipeline_->maxTotalThreadsPerThreadgroup(), 1, 1);

  computeEncoder->dispatchThreads(gridSize, threadGroupSize);
  computeEncoder->endEncoding();
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();
}

void Attention::computeGradientsQKV(MTL::Buffer* gradQBuffer, MTL::Buffer* gradKBuffer, MTL::Buffer* gradVBuffer, MTL::Buffer* inputBuffer, MTL::Buffer* gradInputBuffer, size_t batch_size, size_t sequence_length) {
  // Create command buffer and encoder
  MTL::CommandBuffer* commandBuffer = commandQueue_->commandBuffer();
  MTL::ComputeCommandEncoder* computeEncoder = commandBuffer->computeCommandEncoder();

  // Set pipeline state
  computeEncoder->setComputePipelineState(computeGradientsQKVPipeline_);

  // Set buffers
  computeEncoder->setBuffer(gradQBuffer, 0, 0);
  computeEncoder->setBuffer(gradKBuffer, 0, 1);
  computeEncoder->setBuffer(gradVBuffer, 0, 2);
  computeEncoder->setBuffer(W_query_buffer_, 0, 3);
  computeEncoder->setBuffer(W_key_buffer_, 0, 4);
  computeEncoder->setBuffer(W_value_buffer_, 0, 5);
  computeEncoder->setBuffer(gradInputBuffer, 0, 6);

  // Set constants
  uint input_size = static_cast<uint>(input_size_);
  uint attention_size = static_cast<uint>(attention_size_);
  uint batch_size_uint = static_cast<uint>(batch_size);
  uint sequence_length_uint = static_cast<uint>(sequence_length);
  computeEncoder->setBytes(&input_size, sizeof(uint), 7);
  computeEncoder->setBytes(&attention_size, sizeof(uint), 8);
  computeEncoder->setBytes(&batch_size_uint, sizeof(uint), 9);
  computeEncoder->setBytes(&sequence_length_uint, sizeof(uint), 10);

  // Dispatch threads
  size_t totalThreads = batch_size * sequence_length * input_size_;
  MTL::Size gridSize(totalThreads, 1, 1);
  MTL::Size threadGroupSize = MTL::Size::Make(computeGradientsQKVPipeline_->maxTotalThreadsPerThreadgroup(), 1, 1);

  computeEncoder->dispatchThreads(gridSize, threadGroupSize);
  computeEncoder->endEncoding();
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();
}


