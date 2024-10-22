#include "../../include/layers/activation.h"
#include <algorithm>
#include <cmath>

Activation::Activation(ActivationType type) : type_(type) {}

std::vector<double> Activation::forward(const std::vector<double>& input) {
  input_ = input;
  std::vector<double> output(input.size(), 0.0);
  switch(type_) {
    case ActivationType::ReLU:
      std::transform(input.begin(), input.end(), output.begin(),
                     [](double x) { return x > 0 ? x : 0.0; });
      break;
    case ActivationType::Sigmoid:
      std::transform(input.begin(), input.end(), output.begin(),
                     [](double x) { return 1.0 / (1.0 + std::exp(-x)); });
      break;
    case ActivationType::Tanh:
      std::transform(input.begin(), input.end(), output.begin(),
                     [](double x) { return std::tanh(x); });
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