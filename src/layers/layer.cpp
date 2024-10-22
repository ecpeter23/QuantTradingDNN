#include "../../include/layers/layer.h"
#include "../../include/layers/fully_connected.h"
#include "../../include/layers/activation.h"
#include "../../include/layers/attention.h"
#include "../../include/layers/batch_norm.h"
#include "../../include/layers/conv1d.h"
#include "../../include/layers/dense_activation.h"
#include "../../include/layers/dropout.h"
#include "../../include/layers/lstm.h"
#include "../../include/layers/residual.h"

#include <string>
#include <iostream>

std::unique_ptr<Layer> Layer::load(std::ifstream& ifs) {
  size_t type_length;
  ifs.read(reinterpret_cast<char*>(&type_length), sizeof(size_t));
  std::string layer_type(type_length, ' ');
  ifs.read(&layer_type[0], type_length);

  if(layer_type == "FullyConnected") { return FullyConnected::load(ifs); }
  if(layer_type == "Activation") { return Activation::load(ifs); }
  if (layer_type == "Attention") { return Attention::load(ifs); }
  if (layer_type == "BatchNorm") { return BatchNorm::load(ifs); }
  if (layer_type == "Conv1D") { return Conv1D::load(ifs); }
  if (layer_type == "DenseActivation") { return DenseActivation::load(ifs); }
  if (layer_type == "Dropout") { return Dropout::load(ifs); }
  if (layer_type == "LSTM") { return LSTM::load(ifs); }
  if (layer_type == "Residual") { return Residual::load(ifs); }

  std::cerr << "Unknown layer type: " << layer_type << std::endl;
  return nullptr;
}
