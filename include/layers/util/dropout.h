#ifndef QUANT_TRADING_DNN_DROPOUT_H
#define QUANT_TRADING_DNN_DROPOUT_H

#include "layers/layer.h"
#include <random>

class Dropout : public Layer {
    double dropout_rate_;
    std::vector<double> mask_;  // Stores the mask used during the forward pass

public:
    // Constructor to set the dropout rate
    Dropout(double rate) : dropout_rate_(rate) {
      if (rate < 0.0 || rate > 1.0) {
        throw std::invalid_argument("Dropout: Rate must be between 0 and 1");
      }
    }

    Tensor forward(const Tensor& input) override {
      if (auto vec = std::get_if<std::vector<double>>(&input)) {
        if (is_training_) {
          mask_.resize(vec->size());
          std::random_device rd;
          std::mt19937 gen(rd());
          std::bernoulli_distribution dist(1.0 - dropout_rate_);

          std::vector<double> output(vec->size());
          for (size_t i = 0; i < vec->size(); ++i) {
            mask_[i] = dist(gen);  // Generate mask (0 or 1)
            output[i] = (*vec)[i] * mask_[i];  // Apply mask
          }
          return output;
        } else {
          // During inference, scale the output by (1 - dropout_rate)
          std::vector<double> output(vec->size());
          for (size_t i = 0; i < vec->size(); ++i) {
            output[i] = (*vec)[i] * (1.0 - dropout_rate_);
          }
          return output;
        }
      }
      throw std::runtime_error("Dropout: Unsupported input shape");
    }

    Tensor backward(const Tensor& gradOutput) override {
      if (auto vec = std::get_if<std::vector<double>>(&gradOutput)) {
        std::vector<double> grad(vec->size());
        for (size_t i = 0; i < vec->size(); ++i) {
          grad[i] = (*vec)[i] * mask_[i];  // Apply the same mask
        }
        return grad;
      }
      throw std::runtime_error("Dropout: Unsupported gradient shape");
    }

    void updateWeights(double /*learningRate*/) override {}

    // TODO: Implement save and load functions
    void save(std::ofstream& /*ofs*/) const override {}
    static std::unique_ptr<Layer> load(std::ifstream& /*ifs*/) {
      throw std::runtime_error("Dropout: Load function not implemented");
    }

    [[nodiscard]] std::string layerType() const override {
      return "Dropout";
    }
};

#endif //QUANT_TRADING_DNN_DROPOUT_H
