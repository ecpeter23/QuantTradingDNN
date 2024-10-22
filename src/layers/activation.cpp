#include "../../include/layers/activation.h"
#include <algorithm>
#include <Accelerate/Accelerate.h>
#include <cmath>

Activation::Activation(ActivationType type) : type_(type) {}

std::vector<double> Activation::forward(const std::vector<double>& input) {
  // Assume input is a flattened 2D vector: features x batch_size
  // For example, for 3 features and batch_size 2: [f1s1, f1s2, f2s1, f2s2, f3s1, f3s2]
  input_ = input;
  std::vector<double> output(input.size(), 0.0);


  switch(type_) {
    case ActivationType::ReLU:
      vDSP_vthrD(input.data(), 1, nullptr, output.data(), 1, input.size());
      break;
    case ActivationType::Sigmoid:
      std::transform(input.begin(), input.end(), output.begin(), [](double x) {
          return 1.0 / (1.0 + std::exp(-x));
      });
      break;
    case ActivationType::Tanh:
      vvtanh(output.data(), input.data(), reinterpret_cast<const int *>(input.size()));
      break;
  }
  return output;
}

std::vector<double> Activation::backward(const std::vector<double>& gradOutput) {
  std::vector<double> gradInput(input_.size(), 0.0);
  switch(type_) {
    case ActivationType::ReLU:
      for(size_t i = 0; i < input_.size(); ++i) {
        gradInput[i] = input_[i] > 0 ? gradOutput[i] : 0.0;
      }
      break;
    case ActivationType::Sigmoid:
      for(size_t i = 0; i < input_.size(); ++i) {
        double sigmoid = 1.0 / (1.0 + std::exp(-input_[i]));
        gradInput[i] = gradOutput[i] * sigmoid * (1 - sigmoid);
      }
      break;
    case ActivationType::Tanh:
      for(size_t i = 0; i < input_.size(); ++i) {
        double tanh_val = std::tanh(input_[i]);
        gradInput[i] = gradOutput[i] * (1 - tanh_val * tanh_val);
      }
      break;
  }
  return gradInput;
}

// Implement save
void Activation::save(std::ofstream& ofs) const {
  // Save layer type identifier
  std::string layer_type = "Activation";
  size_t type_length = layer_type.size();
  ofs.write(reinterpret_cast<const char*>(&type_length), sizeof(size_t));
  ofs.write(layer_type.c_str(), type_length);

  // Save activation type
  int act_type = static_cast<int>(type_);
  ofs.write(reinterpret_cast<const char*>(&act_type), sizeof(int));
}

// Implement load
std::unique_ptr<Layer> Activation::load(std::ifstream& ifs) {
  int act_type_int;
  ifs.read(reinterpret_cast<char*>(&act_type_int), sizeof(int));
  ActivationType act_type = static_cast<ActivationType>(act_type_int);
  return std::make_unique<Activation>(act_type);
}