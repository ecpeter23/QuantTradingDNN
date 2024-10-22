#include "layers/dense_activation.h"
#include <cmath>
#include <random>
#include <iostream>

// Constructor: Initializes weights, biases, and gradients
DenseActivation::DenseActivation(size_t input_size, size_t output_size, ActivationType activation)
        : input_size_(input_size), output_size_(output_size), activation_(activation) {
  // Initialize weights and biases with small random values
  std::mt19937 gen(std::random_device{}());
  std::uniform_real_distribution<> dist(-0.1, 0.1);

  weights_.resize(output_size_, std::vector<double>(input_size_, 0.0));
  biases_.resize(output_size_, 0.0);

  // Initialize weights
  for(auto& row : weights_) {
    for(auto& w : row) {
      w = dist(gen);
    }
  }

  // Initialize biases
  for(auto& b : biases_) {
    b = dist(gen);
  }

  // Initialize gradients
  grad_weights_.resize(output_size_, std::vector<double>(input_size_, 0.0));
  grad_biases_.resize(output_size_, 0.0);
}

// Activation function
double DenseActivation::activate(double x) const {
  switch(activation_) {
    case ActivationType::ReLU:
      return x > 0.0 ? x : 0.0;
    case ActivationType::Sigmoid:
      return 1.0 / (1.0 + std::exp(-x));
    case ActivationType::Tanh:
      return std::tanh(x);
    default:
      return x;
  }
}

// Derivative of activation function
double DenseActivation::activateDerivative(double x) const {
  switch(activation_) {
    case ActivationType::ReLU:
      return x > 0.0 ? 1.0 : 0.0;
    case ActivationType::Sigmoid: {
      double sig = 1.0 / (1.0 + std::exp(-x));
      return sig * (1.0 - sig);
    }
    case ActivationType::Tanh: {
      double tanh_x = std::tanh(x);
      return 1.0 - tanh_x * tanh_x;
    }
    default:
      return 1.0;
  }
}

// Forward pass: Computes linear transformation and applies activation
std::vector<double> DenseActivation::forward(const std::vector<double>& input) {
  // Cache input for backward pass
  input_ = input;

  // Compute linear transformation: linear = W * input + b
  linear_.resize(output_size_, 0.0);
  for(size_t i = 0; i < output_size_; ++i) {
    for(size_t j = 0; j < input_size_; ++j) {
      linear_[i] += weights_[i][j] * input[j];
    }
    linear_[i] += biases_[i];
  }

  // Apply activation function
  std::vector<double> output(output_size_, 0.0);
  for(size_t i = 0; i < output_size_; ++i) {
    output[i] = activate(linear_[i]);
  }

  return output;
}

// Backward pass: Computes gradients w.r.t inputs, weights, and biases
std::vector<double> DenseActivation::backward(const std::vector<double>& gradOutput) {
  // Compute gradients w.r.t linear outputs: dL/dlinear = dL/doutput * activationDerivative(linear)
  std::vector<double> grad_linear(output_size_, 0.0);
  for(size_t i = 0; i < output_size_; ++i) {
    grad_linear[i] = gradOutput[i] * activateDerivative(linear_[i]);
  }

  // Compute gradients w.r.t weights and biases
  for(size_t i = 0; i < output_size_; ++i) {
    for(size_t j = 0; j < input_size_; ++j) {
      grad_weights_[i][j] += grad_linear[i] * input_[j];
    }
    grad_biases_[i] += grad_linear[i];
  }

  // Compute gradients w.r.t inputs: dL/dinput = W^T * dL/dlinear
  std::vector<double> grad_input(input_size_, 0.0);
  for(size_t j = 0; j < input_size_; ++j) {
    for(size_t i = 0; i < output_size_; ++i) {
      grad_input[j] += weights_[i][j] * grad_linear[i];
    }
  }

  return grad_input;
}

// Update weights and biases using accumulated gradients and learning rate
void DenseActivation::updateWeights(double learningRate) {
  // Update weights and reset gradients
  for(size_t i = 0; i < output_size_; ++i) {
    for(size_t j = 0; j < input_size_; ++j) {
      weights_[i][j] -= learningRate * grad_weights_[i][j];
      grad_weights_[i][j] = 0.0; // Reset gradient after update
    }
  }

  // Update biases and reset gradients
  for(size_t i = 0; i < output_size_; ++i) {
    biases_[i] -= learningRate * grad_biases_[i];
    grad_biases_[i] = 0.0; // Reset gradient after update
  }
}

// Save the layer's parameters to a binary file
void DenseActivation::save(std::ofstream& ofs) const {
  // Save layer type identifier
  std::string layer_type = "DenseActivation";
  size_t type_length = layer_type.size();
  ofs.write(reinterpret_cast<const char*>(&type_length), sizeof(size_t));
  ofs.write(layer_type.c_str(), type_length);

  // Save parameters
  ofs.write(reinterpret_cast<const char*>(&input_size_), sizeof(size_t));
  ofs.write(reinterpret_cast<const char*>(&output_size_), sizeof(size_t));
  int activation_int = static_cast<int>(activation_);
  ofs.write(reinterpret_cast<const char*>(&activation_int), sizeof(int));

  // Save weights
  for(const auto& row : weights_) {
    ofs.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));
  }

  // Save biases
  for(const auto& b : biases_) {
    ofs.write(reinterpret_cast<const char*>(&b), sizeof(double));
  }
}

// Load the layer's parameters from a binary file
std::unique_ptr<Layer> DenseActivation::load(std::ifstream& ifs) {
  size_t input_size, output_size;
  ifs.read(reinterpret_cast<char*>(&input_size), sizeof(size_t));
  ifs.read(reinterpret_cast<char*>(&output_size), sizeof(size_t));
  int activation_int;
  ifs.read(reinterpret_cast<char*>(&activation_int), sizeof(int));
  ActivationType activation = static_cast<ActivationType>(activation_int);

  auto layer = std::make_unique<DenseActivation>(input_size, output_size, activation);

  // Load weights
  for(auto& row : layer->weights_) {
    ifs.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
  }

  // Load biases
  for(auto& b : layer->biases_) {
    ifs.read(reinterpret_cast<char*>(&b), sizeof(double));
  }

  return layer;
}
