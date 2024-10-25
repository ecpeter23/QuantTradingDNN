#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include "../../include/layers/activation.h"
#include <Metal/Metal.hpp>
#include <AppKit/AppKit.hpp>
#include <MetalKit/MetalKit.hpp>
#include <Accelerate/Accelerate.h>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <cstring>

// Constructor with parameter initialization
Activation::Activation(ActivationType type, bool useGPU, int axis, int num_linear)
        : type_(type), useGPU_(useGPU),
          axis_(axis),
          num_linear_(num_linear),
          alpha_(0.25), d_alpha_(0.0),
          theta_(1.0), d_theta_(0.0),
          beta_(1.0), d_beta_(0.0) {
  if (type_ == ActivationType::APL) {
    // Initialize APL parameters, e.g., two linear pieces
    apl_params_ = {1.0, 0.0}; // Example initial values: alpha1 and alpha2
    d_apl_params_ = {0.0, 0.0};
  }
  if (type_ == ActivationType::Maxout) {
    // Initialize Maxout weights
    maxout_weights_.resize(num_linear_, 1.0);
    d_maxout_weights_.resize(num_linear_, 0.0);
  }
}

// Forward pass: Choose between CPU and GPU implementations
Layer::Tensor Activation::forward(const Tensor& input) {
  input_ = input;
  if (useGPU_) {
    throw std::runtime_error("GPU-based activation functions not implemented.");
  }
  return applyCPUActivation(input_);
}

// CPU-based activation implementations using Accelerate and manual functions
Layer::Tensor Activation::applyCPUActivation(const Tensor& input) {
  Tensor output;
  switch (type_) {
    case ActivationType::ReLU:
    case ActivationType::LeakyReLU:
    case ActivationType::ParametricReLU:
    case ActivationType::ELU:
    case ActivationType::SELU:
    case ActivationType::Sigmoid:
    case ActivationType::Tanh:
    case ActivationType::HardSigmoid:
    case ActivationType::HardTanh:
    case ActivationType::Softplus:
    case ActivationType::Softsign:
    case ActivationType::Swish:
    case ActivationType::GELU:
    case ActivationType::Mish:
    case ActivationType::ThresholdedReLU:
    case ActivationType::APL:
    case ActivationType::BentIdentity:
    case ActivationType::ESwish:
    case ActivationType::LogSigmoid:
    case ActivationType::Sinc:
    case ActivationType::TanhExp:
    {
      // Flatten the tensor, apply element-wise activation, and reconstruct
      std::vector<double> flat_input = flattenTensor(input);
      std::vector<double> flat_output(flat_input.size(), 0.0);

      int size = static_cast<int>(flat_input.size());

      // Apply element-wise activation using Accelerate
      switch (type_) {
        case ActivationType::ReLU:
          vDSP_vthresD(flat_input.data(), 1, nullptr, flat_output.data(), 1, flat_input.size());
          // Replace negative values with 0
          for (size_t i = 0; i < flat_input.size(); ++i) {
            flat_output[i] = std::max(0.0, flat_input[i]);
          }
          break;
        case ActivationType::LeakyReLU:
          // f(x) = x if x > 0 else 0.01x
          vDSP_vsmulD(flat_input.data(), 1, &alpha_, flat_output.data(), 1, flat_input.size());
          for (size_t i = 0; i < flat_input.size(); ++i) {
            flat_output[i] = flat_input[i] > 0.0 ? flat_input[i] : 0.01 * flat_input[i];
          }
          break;
        case ActivationType::ParametricReLU:
          // f(x) = x if x > 0 else alpha * x
          for (size_t i = 0; i < flat_input.size(); ++i) {
            flat_output[i] = flat_input[i] > 0.0 ? flat_input[i] : alpha_ * flat_input[i];
          }
          break;
        case ActivationType::ELU:
          // f(x) = x if x > 0 else exp(x) - 1
          for (size_t i = 0; i < flat_input.size(); ++i) {
            flat_output[i] = flat_input[i] > 0.0 ? flat_input[i] : std::exp(flat_input[i]) - 1.0;
          }
          break;
        case ActivationType::SELU:
        {
          const double lambda = 1.0507009873554804934193349852946;
          const double alpha = 1.6732632423543772848170429916717;
          for (size_t i = 0; i < flat_input.size(); ++i) {
            flat_output[i] = flat_input[i] > 0.0 ? lambda * flat_input[i]
                                                 : lambda * alpha * (std::exp(flat_input[i]) - 1.0);
          }
        }
          break;
        case ActivationType::Sigmoid:
          flat_output.resize(flat_input.size());

          std::transform(flat_input.begin(), flat_input.end(), flat_output.begin(), sigmoid);
          break;
        case ActivationType::Tanh:
          flat_output.resize(flat_input.size());

          std::transform(flat_input.begin(), flat_input.end(), flat_output.begin(),
                         [](double x) -> double { return std::tanh(x); });
          break;
        case ActivationType::HardSigmoid:
          for (size_t i = 0; i < flat_input.size(); ++i) {
            flat_output[i] = hard_sigmoid(flat_input[i]);
          }
          break;
        case ActivationType::HardTanh:
          for (size_t i = 0; i < flat_input.size(); ++i) {
            flat_output[i] = hard_tanh(flat_input[i]);
          }
          break;
        case ActivationType::Softplus:
          for (size_t i = 0; i < flat_input.size(); ++i) {
            flat_output[i] = softplus(flat_input[i]);
          }
          break;
        case ActivationType::Softsign:
          for (size_t i = 0; i < flat_input.size(); ++i) {
            flat_output[i] = softsign(flat_input[i]);
          }
          break;
        case ActivationType::Swish:
          for (size_t i = 0; i < flat_input.size(); ++i) {
            flat_output[i] = swish(flat_input[i]);
          }
          break;
        case ActivationType::GELU:
          for (size_t i = 0; i < flat_input.size(); ++i) {
            flat_output[i] = gelu(flat_input[i]);
          }
          break;
        case ActivationType::Mish:
          for (size_t i = 0; i < flat_input.size(); ++i) {
            flat_output[i] = mish(flat_input[i]);
          }
          break;
        case ActivationType::ThresholdedReLU:
          for (size_t i = 0; i < flat_input.size(); ++i) {
            flat_output[i] = thresholded_relu(flat_input[i], theta_);
          }
          break;
        case ActivationType::APL:
          // Simple APL with two linear pieces: f(x) = alpha1 * x for x < 0, alpha2 * x for x >=0
          for (size_t i = 0; i < flat_input.size(); ++i) {
            flat_output[i] = flat_input[i] < 0.0 ? apl_params_[0] * flat_input[i] : apl_params_[1] * flat_input[i];
          }
          break;
        case ActivationType::BentIdentity:
          for (size_t i = 0; i < flat_input.size(); ++i) {
            flat_output[i] = (std::sqrt(flat_input[i] * flat_input[i] + 1) - 1) / 2 + flat_input[i];
          }
          break;
        case ActivationType::ESwish:
          for (size_t i = 0; i < flat_input.size(); ++i) {
            flat_output[i] = flat_input[i] * sigmoid(beta_ * flat_input[i]);
          }
          break;
        case ActivationType::LogSigmoid:
          for (size_t i = 0; i < flat_input.size(); ++i) {
            flat_output[i] = std::log(sigmoid(flat_input[i]));
          }
          break;
        case ActivationType::Sinc:
          for (size_t i = 0; i < flat_input.size(); ++i) {
            flat_output[i] = flat_input[i] != 0.0 ? std::sin(flat_input[i]) / flat_input[i] : 1.0;
          }
          break;
        case ActivationType::TanhExp:
          for (size_t i = 0; i < flat_input.size(); ++i) {
            flat_output[i] = std::tanh(flat_input[i]) * std::exp(flat_input[i]);
          }
          break;
        default:
          throw std::invalid_argument("Unsupported Activation Type.");
      }

      output = reshapeTensor(flat_output, getShape(input));
    }
      break;

    case ActivationType::Softmax:
      output = applyCPUSoftmax(input);
      break;

    case ActivationType::Maxout:
      output = applyCPUMaxout(input);
      break;

    default:
      throw std::invalid_argument("Unsupported Activation Type.");
  }

  return output;
}

Layer::Tensor Activation::applyCPUMaxout(const Tensor& input) {
  // Maxout is applied across a specified axis with num_linear_ linear pieces
  // Flatten the tensor, determine the size along the axis, apply Maxout

  // Get tensor shape
  std::vector<size_t> shape = getShape(input);
  size_t total_elements = getTotalElements(input);
  if (axis_ < -int(shape.size()) || axis_ >= int(shape.size())) {
    throw std::invalid_argument("Invalid axis for Maxout.");
  }
  if (axis_ < 0) axis_ += shape.size();

  size_t axis_size = shape[axis_];
  if (axis_size % num_linear_ != 0) {
    throw std::invalid_argument("Axis size must be divisible by num_linear_ for Maxout.");
  }
  size_t group_size = axis_size / num_linear_;

  std::vector<double> flat_input = flattenTensor(input);
  std::vector<double> flat_output;

  // Iterate over all positions except along the specified axis
  size_t outer_size = 1;
  for (size_t i = 0; i < axis_; ++i) {
    outer_size *= shape[i];
  }
  size_t inner_size = 1;
  for (size_t i = axis_ + 1; i < shape.size(); ++i) {
    inner_size *= shape[i];
  }

  flat_output.reserve(outer_size * group_size * inner_size);

  for (size_t o = 0; o < outer_size; ++o) {
    for (size_t g = 0; g < group_size; ++g) {
      for (size_t in = 0; in < inner_size; ++in) {
        double max_val = -std::numeric_limits<double>::infinity();
        int max_idx = -1;
        for (int l = 0; l < num_linear_; ++l) {
          size_t idx = o * axis_size * inner_size + (g + l * group_size) * inner_size + in;
          double val = flat_input[idx] * maxout_weights_[l];
          if (val > max_val) {
            max_val = val;
            max_idx = l;
          }
        }
        flat_output.push_back(max_val);
      }
    }
  }

  Tensor output = reshapeTensor(flat_output, shape);
  return output;
}

Layer::Tensor Activation::applyCPUSoftmax(const Tensor& input) {
  // Softmax is applied along a specified axis
  // Flatten the tensor, determine the size along the axis, apply Softmax per slice

  std::vector<size_t> shape = getShape(input);
  size_t total_elements = getTotalElements(input);
  if (axis_ < -int(shape.size()) || axis_ >= int(shape.size())) {
    throw std::invalid_argument("Invalid axis for Softmax.");
  }
  if (axis_ < 0) axis_ += shape.size();

  size_t softmax_size = shape[axis_];
  size_t outer_size = 1;
  for (size_t i = 0; i < axis_; ++i) {
    outer_size *= shape[i];
  }
  size_t inner_size = 1;
  for (size_t i = axis_ + 1; i < shape.size(); ++i) {
    inner_size *= shape[i];
  }

  std::vector<double> flat_input = flattenTensor(input);
  std::vector<double> flat_output(flat_input.size(), 0.0);

  for (size_t o = 0; o < outer_size; ++o) {
    for (size_t in = 0; in < inner_size; ++in) {
      // Find the start index for this slice
      size_t start = o * softmax_size * inner_size + in;
      // Find the max value for numerical stability
      double max_val = -std::numeric_limits<double>::infinity();
      for (size_t s = 0; s < softmax_size; ++s) {
        double val = flat_input[start + s * inner_size];
        if (val > max_val) {
          max_val = val;
        }
      }
      // Compute exponentials and sum
      double sum = 0.0;
      std::vector<double> exps(softmax_size, 0.0);
      for (size_t s = 0; s < softmax_size; ++s) {
        exps[s] = std::exp(flat_input[start + s * inner_size] - max_val);
        sum += exps[s];
      }
      // Compute softmax
      for (size_t s = 0; s < softmax_size; ++s) {
        flat_output[start + s * inner_size] = exps[s] / sum;
      }
    }
  }

  Tensor output = reshapeTensor(flat_output, shape);
  return output;
}

// Backward pass: Compute gradients
Layer::Tensor Activation::backward(const Tensor& gradOutput) {
  Tensor gradInput;
  switch (type_) {
    case ActivationType::ReLU:
    case ActivationType::LeakyReLU:
    case ActivationType::ParametricReLU:
    case ActivationType::ELU:
    case ActivationType::SELU:
    case ActivationType::Sigmoid:
    case ActivationType::Tanh:
    case ActivationType::HardSigmoid:
    case ActivationType::HardTanh:
    case ActivationType::Softplus:
    case ActivationType::Softsign:
    case ActivationType::Swish:
    case ActivationType::GELU:
    case ActivationType::Mish:
    case ActivationType::ThresholdedReLU:
    case ActivationType::APL:
    case ActivationType::BentIdentity:
    case ActivationType::ESwish:
    case ActivationType::LogSigmoid:
    case ActivationType::Sinc:
    case ActivationType::TanhExp:
    {
      // Flatten the tensors
      std::vector<double> flat_input = flattenTensor(input_);
      std::vector<double> flat_gradOutput = flattenTensor(gradOutput);
      std::vector<double> flat_gradInput(flat_gradOutput.size(), 0.0);

      // Reset parameter gradients
      d_alpha_ = 0.0;
      d_theta_ = 0.0;
      d_beta_ = 0.0;
      std::fill(d_apl_params_.begin(), d_apl_params_.end(), 0.0);
      std::fill(d_maxout_weights_.begin(), d_maxout_weights_.end(), 0.0);

      // Compute gradient based on activation type
      switch (type_) {
        case ActivationType::ReLU:
          for (size_t i = 0; i < flat_input.size(); ++i) {
            flat_gradInput[i] = flat_input[i] > 0.0 ? flat_gradOutput[i] : 0.0;
          }
          break;
        case ActivationType::LeakyReLU:
          for (size_t i = 0; i < flat_input.size(); ++i) {
            flat_gradInput[i] = flat_input[i] > 0.0 ? flat_gradOutput[i] : 0.01 * flat_gradOutput[i];
          }
          break;
        case ActivationType::ParametricReLU:
          for (size_t i = 0; i < flat_input.size(); ++i) {
            if (flat_input[i] > 0.0) {
              flat_gradInput[i] = flat_gradOutput[i];
            } else {
              flat_gradInput[i] = alpha_ * flat_gradOutput[i];
              d_alpha_ += flat_input[i] * flat_gradOutput[i];
            }
          }
          break;
        case ActivationType::ELU:
          for (size_t i = 0; i < flat_input.size(); ++i) {
            flat_gradInput[i] = flat_input[i] > 0.0 ? flat_gradOutput[i] : std::exp(flat_input[i]) * flat_gradOutput[i];
          }
          break;
        case ActivationType::SELU:
        {
          const double lambda = 1.0507009873554804934193349852946;
          const double alpha = 1.6732632423543772848170429916717;
          for (size_t i = 0; i < flat_input.size(); ++i) {
            if (flat_input[i] > 0.0) {
              flat_gradInput[i] = lambda * flat_gradOutput[i];
            } else {
              flat_gradInput[i] = lambda * alpha * std::exp(flat_input[i]) * flat_gradOutput[i];
            }
          }
        }
          break;
        case ActivationType::Sigmoid:
          for (size_t i = 0; i < flat_input.size(); ++i) {
            double s = sigmoid(flat_input[i]);
            flat_gradInput[i] = s * (1.0 - s) * flat_gradOutput[i];
          }
          break;
        case ActivationType::Tanh:
          for (size_t i = 0; i < flat_input.size(); ++i) {
            double t = std::tanh(flat_input[i]);
            flat_gradInput[i] = (1.0 - t * t) * flat_gradOutput[i];
          }
          break;
        case ActivationType::HardSigmoid:
          for (size_t i = 0; i < flat_input.size(); ++i) {
            flat_gradInput[i] = hard_sigmoid_derivative(flat_input[i]) * flat_gradOutput[i];
          }
          break;
        case ActivationType::HardTanh:
          for (size_t i = 0; i < flat_input.size(); ++i) {
            flat_gradInput[i] = hard_tanh_derivative(flat_input[i]) * flat_gradOutput[i];
          }
          break;
        case ActivationType::Softplus:
          for (size_t i = 0; i < flat_input.size(); ++i) {
            flat_gradInput[i] = softplus_derivative(flat_input[i]) * flat_gradOutput[i];
          }
          break;
        case ActivationType::Softsign:
          for (size_t i = 0; i < flat_input.size(); ++i) {
            flat_gradInput[i] = softsign_derivative(flat_input[i]) * flat_gradOutput[i];
          }
          break;
        case ActivationType::Swish:
          for (size_t i = 0; i < flat_input.size(); ++i) {
            double s = sigmoid(flat_input[i]);
            flat_gradInput[i] = (s + flat_input[i] * s * (1.0 - s)) * flat_gradOutput[i];
          }
          break;
        case ActivationType::GELU:
          for (size_t i = 0; i < flat_input.size(); ++i) {
            flat_gradInput[i] = gelu_derivative(flat_input[i]) * flat_gradOutput[i];
          }
          break;
        case ActivationType::Mish:
          for (size_t i = 0; i < flat_input.size(); ++i) {
            flat_gradInput[i] = mish_derivative(flat_input[i]) * flat_gradOutput[i];
          }
          break;
        case ActivationType::ThresholdedReLU:
          for (size_t i = 0; i < flat_input.size(); ++i) {
            flat_gradInput[i] = flat_input[i] > theta_ ? flat_gradOutput[i] : 0.0;
          }
          break;
        case ActivationType::APL:
          // f(x) = alpha1 * x for x < 0, alpha2 * x for x >=0
          for (size_t i = 0; i < flat_input.size(); ++i) {
            if (flat_input[i] < 0.0) {
              flat_gradInput[i] = apl_params_[0] * flat_gradOutput[i];
              d_apl_params_[0] += flat_input[i] * flat_gradOutput[i];
            } else {
              flat_gradInput[i] = apl_params_[1] * flat_gradOutput[i];
              d_apl_params_[1] += flat_input[i] * flat_gradOutput[i];
            }
          }
          break;
        case ActivationType::BentIdentity:
          for (size_t i = 0; i < flat_input.size(); ++i) {
            double sqrt_val = std::sqrt(1.0 + flat_input[i] * flat_input[i]);
            flat_gradInput[i] = (flat_input[i] / sqrt_val + 1.0) * flat_gradOutput[i];
          }
          break;
        case ActivationType::ESwish:
          for (size_t i = 0; i < flat_input.size(); ++i) {
            double s = sigmoid(beta_ * flat_input[i]);
            double ds = beta_ * s * (1.0 - s);
            flat_gradInput[i] = (s + flat_input[i] * ds) * flat_gradOutput[i];
            d_beta_ += flat_input[i] * (1.0 - s) * s * flat_input[i] * flat_gradOutput[i];
          }
          break;
        case ActivationType::LogSigmoid:
          for (size_t i = 0; i < flat_input.size(); ++i) {
            double s = sigmoid(flat_input[i]);
            flat_gradInput[i] = (1.0 - s) * flat_gradOutput[i];
          }
          break;
        case ActivationType::Sinc:
          for (size_t i = 0; i < flat_input.size(); ++i) {
            if (flat_input[i] != 0.0) {
              flat_gradInput[i] = (std::cos(flat_input[i]) / flat_input[i] - std::sin(flat_input[i]) / (flat_input[i] * flat_input[i])) * flat_gradOutput[i];
            } else {
              flat_gradInput[i] = 0.0;
            }
          }
          break;
        case ActivationType::TanhExp:
          for (size_t i = 0; i < flat_input.size(); ++i) {
            double t = std::tanh(flat_input[i]);
            double e = std::exp(flat_input[i]);
            flat_gradInput[i] = (e * (1.0 - t * t) + t * e) * flat_gradOutput[i];
          }
          break;
        default:
          throw std::invalid_argument("Unsupported Activation Type.");
      }

      // Reconstruct the gradient tensor
      gradInput = reshapeTensor(flat_gradInput, getShape(input_));
    }
      break;

    case ActivationType::Softmax:
      // For Softmax, assuming it's combined with cross-entropy loss, gradient is (output - target)
      // Here, we directly pass the gradOutput as gradInput
      gradInput = gradOutput;
      break;

    case ActivationType::Maxout:
      // Maxout gradient computation
    {
      std::vector<size_t> shape = getShape(input_);
      size_t total_elements = getTotalElements(input_);
      if (axis_ < -int(shape.size()) || axis_ >= int(shape.size())) {
        throw std::invalid_argument("Invalid axis for Maxout.");
      }
      if (axis_ < 0) axis_ += shape.size();

      size_t axis_size = shape[axis_];
      if (axis_size % num_linear_ != 0) {
        throw std::invalid_argument("Axis size must be divisible by num_linear_ for Maxout.");
      }
      size_t group_size = axis_size / num_linear_;

      std::vector<double> flat_input = flattenTensor(input_);
      std::vector<double> flat_gradOutput = flattenTensor(gradOutput);
      std::vector<double> flat_gradInput(flat_input.size(), 0.0);

      size_t outer_size = 1;
      for (size_t i = 0; i < axis_; ++i) {
        outer_size *= shape[i];
      }
      size_t inner_size = 1;
      for (size_t i = axis_ + 1; i < shape.size(); ++i) {
        inner_size *= shape[i];
      }

      for (size_t o = 0; o < outer_size; ++o) {
        for (size_t g = 0; g < group_size; ++g) {
          for (size_t in = 0; in < inner_size; ++in) {
            double max_val = -std::numeric_limits<double>::infinity();
            int max_idx = -1;
            for (int l = 0; l < num_linear_; ++l) {
              size_t idx = o * axis_size * inner_size + (g + l * group_size) * inner_size + in;
              double val = flat_input[idx] * maxout_weights_[l];
              if (val > max_val) {
                max_val = val;
                max_idx = l;
              }
            }
            if (max_idx != -1) {
              size_t grad_idx = o * group_size * inner_size + g * inner_size + in;
              size_t input_idx = o * axis_size * inner_size + (g + max_idx * group_size) * inner_size + in;
              flat_gradInput[input_idx] += flat_gradOutput[grad_idx];
              // Accumulate gradient for Maxout weights
              d_maxout_weights_[max_idx] += flat_input[input_idx] * flat_gradOutput[grad_idx];
            }
          }
        }
      }

      gradInput = reshapeTensor(flat_gradInput, shape);
    }
      break;

    default:
      throw std::invalid_argument("Unsupported Activation Type.");
  }

  return gradInput;
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
        d_maxout_weights_[i] = 0.0; // Reset after update
      }
      break;
    default:
      break; /* No learnable parameters for other activations */
  }
}

// Helper functions for CPU-based activations
std::vector<double> Activation::flattenTensor(const Layer::Tensor& tensor) {
  std::vector<double> result;

  std::visit([&result](const auto& arg) {
      using ArgType = std::decay_t<decltype(arg)>;

      // Handle 1D tensor
      if constexpr (std::is_same_v<ArgType, std::vector<double>>) {
        result.insert(result.end(), arg.begin(), arg.end());
      }
        // Handle higher-dimensional tensors by recursion
      else {
        for (const auto& subTensor : arg) {
          auto flattenedSubTensor = Activation::flattenTensor(subTensor);
          result.insert(result.end(), flattenedSubTensor.begin(), flattenedSubTensor.end());
        }
      }
  }, tensor);

  return result;
}

Layer::Tensor Activation::reshapeTensor(const std::vector<double>& flat, const std::vector<size_t>& shape) {
  // Depending on the shape, reconstruct the tensor
  if (shape.empty()) {
    throw std::invalid_argument("Shape cannot be empty.");
  }

  // Handle 1D, 2D, 3D, and 4D tensors
  if (shape.size() == 1) {
    return Layer::Tensor(std::vector<double>(flat));
  } else if (shape.size() == 2) {
    std::vector<std::vector<double>> tensor2D(shape[0], std::vector<double>(shape[1]));
    size_t idx = 0;
    for (size_t i = 0; i < shape[0]; ++i) {
      for (size_t j = 0; j < shape[1]; ++j) {
        tensor2D[i][j] = flat[idx++];
      }
    }
    return Layer::Tensor(tensor2D);
  } else if (shape.size() == 3) {
    std::vector<std::vector<std::vector<double>>> tensor3D(shape[0],
                                                           std::vector<std::vector<double>>(shape[1], std::vector<double>(shape[2])));
    size_t idx = 0;
    for (size_t i = 0; i < shape[0]; ++i) {
      for (size_t j = 0; j < shape[1]; ++j) {
        for (size_t k = 0; k < shape[2]; ++k) {
          tensor3D[i][j][k] = flat[idx++];
        }
      }
    }
    return Layer::Tensor(tensor3D);
  } else if (shape.size() == 4) {
    std::vector<std::vector<std::vector<std::vector<double>>>> tensor4D(shape[0],
                                                                        std::vector<std::vector<std::vector<double>>>(shape[1],
                                                                                                                      std::vector<std::vector<double>>(shape[2], std::vector<double>(shape[3]))));
    size_t idx = 0;
    for (size_t i = 0; i < shape[0]; ++i) {
      for (size_t j = 0; j < shape[1]; ++j) {
        for (size_t k = 0; k < shape[2]; ++k) {
          for (size_t l = 0; l < shape[3]; ++l) {
            tensor4D[i][j][k][l] = flat[idx++];
          }
        }
      }
    }
    return Layer::Tensor(tensor4D);
  } else {
    throw std::invalid_argument("Only 1D to 4D tensors are supported.");
  }
}

std::vector<size_t> Activation::getShape(const Layer::Tensor& tensor) {
  return std::visit([](auto&& arg) -> std::vector<size_t> {
      std::vector<size_t> shape;
      if constexpr (std::is_same_v<decltype(arg), std::vector<double>>) {
        shape = { arg.size() };
      } else if constexpr (std::is_same_v<decltype(arg), std::vector<std::vector<double>>>) {
        shape = { arg.size(), arg.empty() ? 0 : arg[0].size() };
      } else if constexpr (std::is_same_v<decltype(arg), std::vector<std::vector<std::vector<double>>>>) {
        shape = { arg.size(),
                  arg.empty() ? 0 : arg[0].size(),
                  (arg.empty() || arg[0].empty()) ? 0 : arg[0][0].size() };
      } else if constexpr (std::is_same_v<decltype(arg), std::vector<std::vector<std::vector<std::vector<double>>>>>) {
        shape = { arg.size(),
                  arg.empty() ? 0 : arg[0].size(),
                  (arg.empty() || arg[0].empty()) ? 0 : arg[0][0].size(),
                  (arg.empty() || arg[0].empty() || arg[0][0].empty()) ? 0 : arg[0][0][0].size() };
      }
      return shape;
  }, tensor);
}

size_t Activation::getTotalElements(const Layer::Tensor& tensor) {
  std::vector<size_t> shape = getShape(tensor);
  size_t total = 1;
  for (size_t dim : shape) {
    total *= dim;
  }
  return total;
}

Layer::Tensor Activation::reconstructTensor(const std::vector<double>& flat, const Layer::Tensor& original) {
  std::vector<size_t> shape = getShape(original);
  return reshapeTensor(flat, shape);
}

