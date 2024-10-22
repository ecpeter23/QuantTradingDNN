#ifndef QUANT_TRADING_DNN_FLATTEN_H
#define QUANT_TRADING_DNN_FLATTEN_H

#include "layers/layer.h"
#include <vector>
#include <random>

class Flatten : public Layer {
public:
    Tensor forward(const Tensor& input) override {
      if (auto vec = std::get_if<std::vector<std::vector<double>>>(&input)) {
        std::vector<double> flattened;
        for (const auto& row : *vec) {
          flattened.insert(flattened.end(), row.begin(), row.end());
        }
        return flattened;
      }
      throw std::runtime_error("Unsupported input shape for Flatten layer");
    }

    Tensor backward(const Tensor& /*gradOutput*/) override {
      throw std::runtime_error("Flatten layer does not support backward pass");
    }

    void updateWeights(double /*learningRate*/) override {}

    void save(std::ofstream& /*ofs*/) const override {}

    [[nodiscard]] std::string layerType() const override {
      return "Flatten";
    }
};



#endif //QUANT_TRADING_DNN_FLATTEN_H
