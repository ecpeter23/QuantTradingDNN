#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include "../../include/layers/activation.h"
#include "../../include/utils/math.h"
#include <Metal/Metal.hpp>
#include <AppKit/AppKit.hpp>
#include <MetalKit/MetalKit.hpp>
#include <Accelerate/Accelerate.h>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <cstring>

// Constructor with parameter initialization
Activation::Activation(ActivationType type, bool useGPU)
        : type_(type), useGPU_(useGPU),
          alpha_(0.25), d_alpha_(0.0),
          theta_(1.0), d_theta_(0.0),
          beta_(1.0), d_beta_(0.0),
          num_linear_(2) { // Default for Maxout
  if (type_ == ActivationType::APL) {
    // Initialize APL parameters, e.g., two linear pieces
    apl_params_ = {1.0, 0.0}; // Example initial values
    d_apl_params_ = {0.0, 0.0};
  }
  if (type_ == ActivationType::Maxout) {
    // Initialize Maxout weights (simplified)
    maxout_weights_.resize(num_linear_, 1.0);
    d_maxout_weights_.resize(num_linear_, 0.0);
  }
}

// Forward pass: Choose between CPU and GPU implementations
Layer::Tensor Activation::forward(const Tensor& input) {
  input_ = std::get<std::vector<double>>(input);
  if (useGPU_) {
    return applyGPUActivation(input_);
  }
  return applyCPUActivation(input_);
}

// CPU-based activation implementations using vDSP and manual functions
Layer::Tensor Activation::applyCPUActivation(const std::vector<double>& input) {
  std::vector<double> output(input.size(), 0.0);

  switch (type_) {
    case ActivationType::ReLU:
      std::transform(input.begin(), input.end(), output.begin(),
                     [](double x) -> double { return std::max(0.0, x); });
      break;
    case ActivationType::LeakyReLU:
      std::transform(input.begin(), input.end(), output.begin(),
                     [](double x) -> double { return x > 0.0 ? x : 0.01 * x; });
      break;
    case ActivationType::ParametricReLU:
      std::transform(input.begin(), input.end(), output.begin(),
                     [this](double x) -> double { return x > 0.0 ? x : this->alpha_ * x; });
      break;
    case ActivationType::ELU:
      std::transform(input.begin(), input.end(), output.begin(),
                     [](double x) -> double { return x > 0.0 ? x : std::exp(x) - 1.0; });
      break;
    case ActivationType::SELU:
    {
      const double lambda = 1.0507009873554804934193349852946;
      const double alpha = 1.6732632423543772848170429916717;
      std::transform(input.begin(), input.end(), output.begin(),
                     [&](double x) -> double {
                         return x > 0.0 ? lambda * x : lambda * alpha * (std::exp(x) - 1.0);
                     });
    }
      break;
    case ActivationType::Sigmoid:
      std::transform(input.begin(), input.end(), output.begin(), sigmoid);
      break;
    case ActivationType::Tanh:
      std::transform(input.begin(), input.end(), output.begin(),
                     [](double x) -> double { return std::tanh(x); });
      break;
    case ActivationType::HardSigmoid:
      std::transform(input.begin(), input.end(), output.begin(), hard_sigmoid);
      break;
    case ActivationType::HardTanh:
      std::transform(input.begin(), input.end(), output.begin(), hard_tanh);
      break;
    case ActivationType::Softmax:
    {
      double max_val = *std::max_element(input.begin(), input.end());
      double sum = 0.0;
      std::vector<double> exps(input.size());
      for (size_t i = 0; i < input.size(); ++i) {
        exps[i] = std::exp(input[i] - max_val);
        sum += exps[i];
      }
      std::transform(exps.begin(), exps.end(), output.begin(),
                     [&](double x) -> double { return x / sum; });
    }
      break;
    case ActivationType::Softplus:
      std::transform(input.begin(), input.end(), output.begin(), softplus);
      break;
    case ActivationType::Softsign:
      std::transform(input.begin(), input.end(), output.begin(), softsign);
      break;
    case ActivationType::Swish:
      std::transform(input.begin(), input.end(), output.begin(), swish);
      break;
    case ActivationType::GELU:
      std::transform(input.begin(), input.end(), output.begin(), gelu);
      break;
    case ActivationType::Maxout:
    {
      // Assuming input size is divisible by num_linear_
      size_t group_size = input.size() / num_linear_;
      for (size_t i = 0; i < group_size; ++i) {
        double max_val = -std::numeric_limits<double>::infinity();
        for (int j = 0; j < num_linear_; ++j) {
          double val = input[i + j * group_size] * maxout_weights_[j];
          if (val > max_val) {
            max_val = val;
          }
        }
        output[i] = max_val;
      }
    }
      break;
    case ActivationType::Mish:
      std::transform(input.begin(), input.end(), output.begin(), mish);
      break;
    case ActivationType::ThresholdedReLU:
      std::transform(input.begin(), input.end(), output.begin(),
                     [this](double x) -> double { return x > this->theta_ ? x : 0.0; });
      break;
    case ActivationType::APL:
      // Simple APL with two linear pieces: f(x) = alpha1 * x for x < 0, alpha2 * x for x >=0
      std::transform(input.begin(), input.end(), output.begin(),
                     [&](double x) -> double {
                         return x < 0.0 ? apl_params_[0] * x : apl_params_[1] * x;
                     });
      break;
    case ActivationType::BentIdentity:
      std::transform(input.begin(), input.end(), output.begin(),
                     [](double x) -> double { return (std::sqrt(x * x + 1) - 1) / 2 + x; });
      break;
    case ActivationType::ESwish:
      std::transform(input.begin(), input.end(), output.begin(),
                     [this](double x) -> double { return x * sigmoid(this->beta_ * x); });
      break;
    case ActivationType::LogSigmoid:
      std::transform(input.begin(), input.end(), output.begin(),
                     [](double x) -> double { return std::log(sigmoid(x)); });
      break;
    case ActivationType::Sinc:
      std::transform(input.begin(), input.end(), output.begin(),
                     [](double x) -> double { return x != 0.0 ? std::sin(x) / x : 1.0; });
      break;
    case ActivationType::TanhExp:
      std::transform(input.begin(), input.end(), output.begin(),
                     [](double x) -> double { return std::tanh(x) * std::exp(x); });
      break;
    default:
      throw std::invalid_argument("Unsupported Activation Type.");
  }

  return Tensor(output);
}

Layer::Tensor Activation::applyGPUActivation(const std::vector<double>& input) {
  throw std::runtime_error("GPU-based activation functions not implemented.");
}

// Backward pass: Compute gradients
Layer::Tensor Activation::backward(const Tensor& gradOutput) {
  std::vector<double> grad = std::get<std::vector<double>>(gradOutput);
  std::vector<double> gradInput(grad.size(), 0.0);

  // Reset parameter gradients
  d_alpha_ = 0.0;
  d_theta_ = 0.0;
  d_beta_ = 0.0;
  std::fill(d_apl_params_.begin(), d_apl_params_.end(), 0.0);
  std::fill(d_maxout_weights_.begin(), d_maxout_weights_.end(), 0.0);

  switch (type_) {
    case ActivationType::ReLU:
      for (size_t i = 0; i < grad.size(); ++i) {
        gradInput[i] = input_[i] > 0.0 ? grad[i] : 0.0;
      }
      break;
    case ActivationType::LeakyReLU:
      for (size_t i = 0; i < grad.size(); ++i) {
        gradInput[i] = input_[i] > 0.0 ? grad[i] : 0.01 * grad[i];
      }
      break;
    case ActivationType::ParametricReLU:
      for (size_t i = 0; i < grad.size(); ++i) {
        if (input_[i] > 0.0) {
          gradInput[i] = grad[i];
        } else {
          gradInput[i] = this->alpha_ * grad[i];
          d_alpha_ += input_[i] * grad[i];
        }
      }
      break;
    case ActivationType::ELU:
      for (size_t i = 0; i < grad.size(); ++i) {
        gradInput[i] = input_[i] > 0.0 ? grad[i] : std::exp(input_[i]) * grad[i];
      }
      break;
    case ActivationType::SELU:
    {
      const double lambda = 1.0507009873554804934193349852946;
      const double alpha = 1.6732632423543772848170429916717;
      for (size_t i = 0; i < grad.size(); ++i) {
        if (input_[i] > 0.0) {
          gradInput[i] = lambda * grad[i];
        } else {
          gradInput[i] = lambda * alpha * std::exp(input_[i]) * grad[i];
        }
      }
    }
      break;
    case ActivationType::Sigmoid:
      for (size_t i = 0; i < grad.size(); ++i) {
        double s = sigmoid(input_[i]);
        gradInput[i] = s * (1.0 - s) * grad[i];
      }
      break;
    case ActivationType::Tanh:
      for (size_t i = 0; i < grad.size(); ++i) {
        double t = std::tanh(input_[i]);
        gradInput[i] = (1.0 - t * t) * grad[i];
      }
      break;
    case ActivationType::HardSigmoid:
      for (size_t i = 0; i < grad.size(); ++i) {
        gradInput[i] = hard_sigmoid_derivative(input_[i]) * grad[i];
      }
      break;
    case ActivationType::HardTanh:
      for (size_t i = 0; i < grad.size(); ++i) {
        gradInput[i] = hard_tanh_derivative(input_[i]) * grad[i];
      }
      break;
    case ActivationType::Softmax:
    {
      // Assuming cross-entropy loss, gradient is (output - target)
      // If not, a full Jacobian is needed
      // Here, we assume it's combined with cross-entropy
      gradInput = grad; // Placeholder
    }
      break;
    case ActivationType::Softplus:
      for (size_t i = 0; i < grad.size(); ++i) {
        gradInput[i] = softplus_derivative(input_[i]) * grad[i];
      }
      break;
    case ActivationType::Softsign:
      for (size_t i = 0; i < grad.size(); ++i) {
        gradInput[i] = softsign_derivative(input_[i]) * grad[i];
      }
      break;
    case ActivationType::Swish:
      for (size_t i = 0; i < grad.size(); ++i) {
        double s = sigmoid(input_[i]);
        gradInput[i] = (s + input_[i] * s * (1.0 - s)) * grad[i];
      }
      break;
    case ActivationType::GELU:
      for (size_t i = 0; i < grad.size(); ++i) {
        gradInput[i] = gelu_derivative(input_[i]) * grad[i];
      }
      break;
    case ActivationType::Maxout:
    {
      // Simplified Maxout gradient: pass gradient to the max input
      size_t group_size = input_.size() / num_linear_;
      for (size_t i = 0; i < group_size; ++i) {
        double max_val = -std::numeric_limits<double>::infinity();
        int max_index = -1;
        for (int j = 0; j < num_linear_; ++j) {
          double val = input_[i + j * group_size] * maxout_weights_[j];
          if (val > max_val) {
            max_val = val;
            max_index = j;
          }
        }
        if (max_index != -1) {
          gradInput[i + max_index * group_size] += grad[i];
        }
      }
    }
      break;
    case ActivationType::Mish:
      for (size_t i = 0; i < grad.size(); ++i) {
        gradInput[i] = mish_derivative(input_[i]) * grad[i];
      }
      break;
    case ActivationType::ThresholdedReLU:
      for (size_t i = 0; i < grad.size(); ++i) {
        gradInput[i] = input_[i] > this->theta_ ? grad[i] : 0.0;
      }
      break;
    case ActivationType::APL:
      // f(x) = alpha1 * x for x < 0, alpha2 * x for x >=0
      for (size_t i = 0; i < grad.size(); ++i) {
        if (input_[i] < 0.0) {
          gradInput[i] = apl_params_[0] * grad[i];
          d_apl_params_[0] += input_[i] * grad[i];
        } else {
          gradInput[i] = apl_params_[1] * grad[i];
          d_apl_params_[1] += input_[i] * grad[i];
        }
      }
      break;
    case ActivationType::BentIdentity:
      for (size_t i = 0; i < grad.size(); ++i) {
        double sqrt_val = std::sqrt(1.0 + input_[i] * input_[i]);
        gradInput[i] = (input_[i] / sqrt_val + 1.0) * grad[i];
      }
      break;
    case ActivationType::ESwish:
      for (size_t i = 0; i < grad.size(); ++i) {
        double s = sigmoid(beta_ * input_[i]);
        double ds = beta_ * s * (1.0 - s);
        gradInput[i] = (s + input_[i] * ds) * grad[i];
        d_beta_ += input_[i] * (1.0 - s) * s * input_[i] * grad[i];
      }
      break;
    case ActivationType::LogSigmoid:
      for (size_t i = 0; i < grad.size(); ++i) {
        double s = sigmoid(input_[i]);
        gradInput[i] = (1.0 - s) * grad[i];
      }
      break;
    case ActivationType::Sinc:
      for (size_t i = 0; i < grad.size(); ++i) {
        if (input_[i] != 0.0) {
          gradInput[i] = (std::cos(input_[i]) / input_[i] - std::sin(input_[i]) / (input_[i] * input_[i])) * grad[i];
        } else {
          gradInput[i] = 0.0;
        }
      }
      break;
    case ActivationType::TanhExp:
      for (size_t i = 0; i < grad.size(); ++i) {
        double t = std::tanh(input_[i]);
        double e = std::exp(input_[i]);
        gradInput[i] = (e * (1.0 - t * t) + t * e) * grad[i];
      }
      break;
    default:
      throw std::invalid_argument("Unsupported Activation Type.");
  }

  return Tensor(gradInput);
}

// Update learnable parameters based on gradients
void Activation::updateWeights(double learningRate) {
  switch (type_) {
    case ActivationType::ParametricReLU:
      alpha_ -= learningRate * d_alpha_;
      break;
    case ActivationType::ThresholdedReLU:
      theta_ -= learningRate * d_theta_;
      break;
    case ActivationType::ESwish:
      beta_ -= learningRate * d_beta_;
      break;
    case ActivationType::APL:
      for (size_t i = 0; i < apl_params_.size(); ++i) {
        apl_params_[i] -= learningRate * d_apl_params_[i];
      }
      break;
    case ActivationType::Maxout:
      for (size_t i = 0; i < maxout_weights_.size(); ++i) {
        maxout_weights_[i] -= learningRate * d_maxout_weights_[i];
      }
      break;
    default:
      // No parameters to update
      break;
  }
}
