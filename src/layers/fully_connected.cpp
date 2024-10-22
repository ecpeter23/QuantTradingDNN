#include "../../include/layers/fully_connected.h"
#include "../../include/utils/math.h"
#include <random>
#include <Accelerate/Accelerate.h>
#include <stdexcept>

FullyConnected::FullyConnected(int inputSize, int outputSize)
        : inputSize_(inputSize), outputSize_(outputSize) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dist(-0.1, 0.1);

  weights_.resize(outputSize_, std::vector<double>(inputSize_));
  for (auto& row : weights_) {
    for (auto& val : row) {
      val = dist(gen);
    }
  }
  biases_.resize(outputSize_, 0.0);
  gradWeights_.resize(outputSize_, std::vector<double>(inputSize_, 0.0));
  gradBiases_.resize(outputSize_, 0.0);
}


// Forward pass: Y = X * W + b
Layer::Tensor FullyConnected::forward(const Layer::Tensor& input) {
  input_ = std::get<std::vector<double>>(input);  // Store input for backprop
  std::vector<double> output(outputSize_, 0.0);

  // Perform matrix multiplication using Accelerate (BLAS)
  cblas_dgemv(CblasRowMajor, CblasNoTrans, outputSize_, inputSize_,
              1.0, &weights_[0][0], inputSize_, input_.data(), 1, 1.0,
              output.data(), 1);

  // Add biases to the output
  for (int i = 0; i < outputSize_; ++i) {
    output[i] += biases_[i];
  }
  return output;
}

// Backward pass: Compute gradients of weights and biases
Layer::Tensor FullyConnected::backward(const Layer::Tensor& gradOutput) {
  std::vector<double> grad = std::get<std::vector<double>>(gradOutput);

  // Compute gradient of weights: dW = X^T * dY
  cblas_dger(CblasRowMajor, outputSize_, inputSize_, 1.0,
             grad.data(), 1, input_.data(), 1, &gradWeights_[0][0], inputSize_);

  // Compute gradient of biases: db = sum(dY)
  for (int i = 0; i < outputSize_; ++i) {
    gradBiases_[i] += grad[i];
  }

  // Compute gradient with respect to input: dX = dY * W^T
  std::vector<double> gradInput(inputSize_, 0.0);
  cblas_dgemv(CblasRowMajor, CblasTrans, outputSize_, inputSize_,
              1.0, &weights_[0][0], inputSize_, grad.data(), 1, 0.0,
              gradInput.data(), 1);
  return gradInput;
}

// Update weights: W = W - eta * dW, b = b - eta * db
void FullyConnected::updateWeights(double learningRate) {
  for (int i = 0; i < outputSize_; ++i) {
    for (int j = 0; j < inputSize_; ++j) {
      weights_[i][j] -= learningRate * gradWeights_[i][j];
      gradWeights_[i][j] = 0.0;  // Reset gradient after update
    }
    biases_[i] -= learningRate * gradBiases_[i];
    gradBiases_[i] = 0.0;  // Reset gradient after update
  }
}

// Implement save
void FullyConnected::save(std::ofstream& ofs) const {
  // Save layer type identifier
  std::string layer_type = "FullyConnected";
  size_t type_length = layer_type.size();
  ofs.write(reinterpret_cast<const char*>(&type_length), sizeof(size_t));
  ofs.write(layer_type.c_str(), type_length);

  // Save input and output sizes
  ofs.write(reinterpret_cast<const char*>(&inputSize_), sizeof(int));
  ofs.write(reinterpret_cast<const char*>(&outputSize_), sizeof(int));

  // Save weights
  for(const auto& row : weights_) {
    ofs.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));
  }

  // Save biases
  ofs.write(reinterpret_cast<const char*>(biases_.data()), biases_.size() * sizeof(double));
}

// Implement load (static factory method)
std::unique_ptr<Layer> FullyConnected::load(std::ifstream& ifs) {
  int input_size, output_size;
  ifs.read(reinterpret_cast<char*>(&input_size), sizeof(int));
  ifs.read(reinterpret_cast<char*>(&output_size), sizeof(int));

  auto layer = std::make_unique<FullyConnected>(input_size, output_size);

  // Load weights
  for(auto& row : layer->weights_) {
    ifs.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
  }

  // Load biases
  ifs.read(reinterpret_cast<char*>(layer->biases_.data()), layer->biases_.size() * sizeof(double));

  return layer;
}
