#include "layers/dense_activation.h"
#include <cmath>
#include <random>
#include <iostream>

DenseActivation::DenseActivation(size_t input_size, size_t output_size, ActivationType activation)
        : input_size_(input_size), output_size_(output_size), activation_(activation) {
  // Initialize weights and biases with small random values
  std::mt19937 gen(std::random_device{}());
  std::uniform_real_distribution<> dist(-0.1, 0.1);

  weights_.resize(output_size_, std::vector<double>(input_size_, 0.0));
  biases_.resize(output_size_, 0.0);

  for(auto& row : weights_) {
    for(auto& w : row) {
      w = dist(gen);
    }
  }

  for(auto& b : biases_) {
    b = dist(gen);
  }
}

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

double DenseActivation::activateDerivative(double x) const {
  switch(activation_) {
    case ActivationType::ReLU:
      return x > 0.0 ? 1.0 : 0.0;
    case ActivationType::Sigmoid:
      double sig = 1.0 / (1.0 + std::exp(-x));
      return sig * (1.0 - sig);
    case ActivationType::Tanh:
      double tanh_x = std::tanh(x);
      return 1.0 - tanh_x * tanh_x;
    default:
      return 1.0;
  }
}

std::vector<double> DenseActivation::forward(const std::vector<double>& input) {
  // Compute linear transformation
  std::vector<double> linear(output_size_, 0.0);
  for(size_t i = 0; i < output_size_; ++i) {
    for(size_t j = 0; j < input_size_; ++j) {
      linear[i] += weights_[i][j] * input[j];
    }
    linear[i] += biases_[i];
  }

  // Apply activation function
  std::vector<double> output(output_size_, 0.0);
  for(size_t i = 0; i < output_size_; ++i) {
    output[i] = activate(linear[i]);
  }

  return output;
}

std::vector<double> DenseActivation::backward(const std::vector<double>& gradOutput) {
  // Placeholder implementation
  // Compute gradients w.r.t weights, biases, and input
  // Apply activation derivative
  return std::vector<double>(input_size_, 0.0);
}

void DenseActivation::updateWeights(double learningRate) {
  // Placeholder implementation
  // Update weights and biases using computed gradients
}

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

  // Save weights and biases
  for(const auto& row : weights_) {
    ofs.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));
  }
  for(const auto& b : biases_) {
    ofs.write(reinterpret_cast<const char*>(&b), sizeof(double));
  }
}

std::unique_ptr<Layer> DenseActivation::load(std::ifstream& ifs) {
  size_t input_size, output_size;
  ifs.read(reinterpret_cast<char*>(&input_size), sizeof(size_t));
  ifs.read(reinterpret_cast<char*>(&output_size), sizeof(size_t));
  int activation_int;
  ifs.read(reinterpret_cast<char*>(&activation_int), sizeof(int));
  ActivationType activation = static_cast<ActivationType>(activation_int);

  auto layer = std::make_unique<DenseActivation>(input_size, output_size, activation);

  // Load weights and biases
  for(auto& row : layer->weights_) {
    ifs.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
  }
  for(auto& b : layer->biases_) {
    ifs.read(reinterpret_cast<char*>(&b), sizeof(double));
  }

  return layer;
}