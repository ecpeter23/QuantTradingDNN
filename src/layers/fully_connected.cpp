#include "../../include/layers/fully_connected.h"
#include "../../include/utils/math.h"
#include <random>

FullyConnected::FullyConnected(int inputSize, int outputSize)
        : inputSize_(inputSize), outputSize_(outputSize),
          weights_(outputSize, std::vector<double>(inputSize)),
          biases_(outputSize, 0.0),
          gradWeights_(outputSize, std::vector<double>(inputSize, 0.0)),
          gradBiases_(outputSize, 0.0) {
  // Initialize weights with small random values
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-0.01, 0.01);
  for(auto &row : weights_) {
    for(auto &w : row) {
      w = dis(gen);
    }
  }
}

std::vector<double> FullyConnected::forward(const std::vector<double>& input) {
  input_ = input;
  // Perform matrix multiplication: output = weights * input + biases
  std::vector<double> output = matMul(weights_, input);
  for(int i = 0; i < output.size(); ++i) {
    output[i] += biases_[i];
  }
  return output;
}

std::vector<double> FullyConnected::backward(const std::vector<double>& gradOutput) {
  // Compute gradients w.r. to weights and biases
  for(int i = 0; i < outputSize_; ++i) {
    for(int j = 0; j < inputSize_; ++j) {
      gradWeights_[i][j] += gradOutput[i] * input_[j];
    }
    gradBiases_[i] += gradOutput[i];
  }

  // Compute gradient w.r. to input for previous layer
  std::vector<double> gradInput(inputSize_, 0.0);
  for(int i = 0; i < inputSize_; ++i) {
    for(int j = 0; j < outputSize_; ++j) {
      gradInput[i] += weights_[j][i] * gradOutput[j];
    }
  }
  return gradInput;
}

void FullyConnected::updateWeights(double learningRate) {
  // Update weights and biases using accumulated gradients
  for(int i = 0; i < outputSize_; ++i) {
    for(int j = 0; j < inputSize_; ++j) {
      weights_[i][j] -= learningRate * gradWeights_[i][j];
      gradWeights_[i][j] = 0.0; // Reset gradient
    }
    biases_[i] -= learningRate * gradBiases_[i];
    gradBiases_[i] = 0.0; // Reset gradient
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
