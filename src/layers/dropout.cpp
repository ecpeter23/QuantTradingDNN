#include "layers/dropout.h"
Dropout::Dropout(double rate)
        : rate_(rate), gen_(std::random_device{}()), dist_(1.0 - rate) {}

std::vector<double> Dropout::forward(const std::vector<double>& input) {
  mask_.resize(input.size());
  std::vector<double> output(input.size());
  for(size_t i = 0; i < input.size(); ++i) {
    mask_[i] = dist_(gen_);
    output[i] = mask_[i] ? input[i] / (1.0 - rate_) : 0.0;
  }
  return output;
}

std::vector<double> Dropout::backward(const std::vector<double>& gradOutput) {
  std::vector<double> gradInput(gradOutput.size());
  for(size_t i = 0; i < gradOutput.size(); ++i) {
    gradInput[i] = mask_[i] ? gradOutput[i] / (1.0 - rate_) : 0.0;
  }
  return gradInput;
}

void Dropout::save(std::ofstream& ofs) const {
  // Save layer type identifier
  std::string layer_type = "Dropout";
  size_t type_length = layer_type.size();
  ofs.write(reinterpret_cast<const char*>(&type_length), sizeof(size_t));
  ofs.write(layer_type.c_str(), type_length);

  // Save dropout rate
  ofs.write(reinterpret_cast<const char*>(&rate_), sizeof(double));
}

std::unique_ptr<Layer> Dropout::load(std::ifstream& ifs) {
  double rate;
  ifs.read(reinterpret_cast<char*>(&rate), sizeof(double));
  return std::make_unique<Dropout>(rate);
}
