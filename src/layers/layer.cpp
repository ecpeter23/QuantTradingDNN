#include "../../include/layers/layer.h"
#include "../../include/layers/fully_connected.h"
#include "../../include/layers/activation.h"
#include <string>
#include <iostream>

std::unique_ptr<Layer> Layer::load(std::ifstream& ifs) {
  // Read layer type identifier
  size_t type_length;
  ifs.read(reinterpret_cast<char*>(&type_length), sizeof(size_t));
  std::string layer_type(type_length, ' ');
  ifs.read(&layer_type[0], type_length);

  if(layer_type == "FullyConnected") {
    return FullyConnected::load(ifs);
  }
  else if(layer_type == "Activation") {
    return Activation::load(ifs);
  }
  else {
    std::cerr << "Unknown layer type: " << layer_type << std::endl;
    return nullptr;
  }
}
