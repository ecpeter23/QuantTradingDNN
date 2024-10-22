#ifndef QUANT_TRADING_DNN_RESHAPE_H
#define QUANT_TRADING_DNN_RESHAPE_H

#include "../layer.h"
#include <stdexcept>

class Reshape : public Layer {
    int new_rows_, new_cols_;

public:
    // Constructor to define the target shape
    Reshape(int new_rows, int new_cols) : new_rows_(new_rows), new_cols_(new_cols) {}

    // Forward pass: Reshape input into the specified dimensions
    Tensor forward(const Tensor& input) override {
      if (auto vec = std::get_if<std::vector<double>>(&input)) {
        if (vec->size() != new_rows_ * new_cols_) {
          throw std::runtime_error("Reshape: Input size does not match target shape");
        }

        std::vector<std::vector<double>> reshaped(new_rows_, std::vector<double>(new_cols_));
        for (int i = 0; i < new_rows_; ++i) {
          for (int j = 0; j < new_cols_; ++j) {
            reshaped[i][j] = (*vec)[i * new_cols_ + j];
          }
        }
        return reshaped;
      }
      throw std::runtime_error("Reshape: Unsupported input shape");
    }

    Tensor backward(const Tensor& gradOutput) override {
      // For Reshape, backward pass is just flattening the gradient
      if (auto mat = std::get_if<std::vector<std::vector<double>>>(&gradOutput)) {
        std::vector<double> flattened;
        for (const auto& row : *mat) {
          flattened.insert(flattened.end(), row.begin(), row.end());
        }
        return flattened;
      }
      throw std::runtime_error("Reshape: Unsupported gradient shape");
    }

    void updateWeights(double /*learningRate*/) override {}

    void save(std::ofstream& /*ofs*/) const override {}

    [[nodiscard]] std::string layerType() const override {
      return "Reshape";
    }
};

#endif //QUANT_TRADING_DNN_RESHAPE_H
