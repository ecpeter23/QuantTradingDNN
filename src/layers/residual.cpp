#include "layers/residual.h"
#include <iostream>

Residual::Residual(std::shared_ptr<Layer> subLayer)
        : subLayer_(subLayer) {}

std::vector<double> Residual::forward(const std::vector<double>& input) {
  std::vector<double> subLayerOutput = subLayer_->forward(input);
  std::vector<double> output(subLayerOutput.size());
  for(size_t i = 0; i < subLayerOutput.size(); ++i) {
    output[i] = subLayerOutput[i] + input[i];
  }
  return output;
}

std::vector<double> Residual::backward(const std::vector<double>& gradOutput) {
  // Gradient flows to both the subLayer and the input
  std::vector<double> gradSubLayer = subLayer_->backward(gradOutput);
  std::vector<double> gradInput(gradOutput.size(), 0.0);
  for(size_t i = 0; i < gradOutput.size(); ++i) {
    gradInput[i] += gradOutput[i];
  }
  return gradInput;
}

void Residual::updateWeights(double learningRate) {
  subLayer_->updateWeights(learningRate);
}

void Residual::save(std::ofstream& ofs) const {
  // Save layer type identifier
  std::string layer_type = "Residual";
  size_t type_length = layer_type.size();
  ofs.write(reinterpret_cast<const char*>(&type_length), sizeof(size_t));
  ofs.write(layer_type.c_str(), type_length);

  // Save the subLayer
  subLayer_->save(ofs);
}

std::unique_ptr<Layer> Residual::load(std::ifstream& ifs) {
  // Load the subLayer
  std::unique_ptr<Layer> subLayer = Layer::load(ifs);
  if(!subLayer) {
    std::cerr << "Failed to load subLayer for Residual connection." << std::endl;
    return nullptr;
  }
  return std::make_unique<Residual>(std::shared_ptr<Layer>(std::move(subLayer)));
}
