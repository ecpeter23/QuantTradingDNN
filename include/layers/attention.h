//
// Created by Eli Peter on 10/22/24.
//

#ifndef QUANT_TRADING_DNN_ATTENTION_H
#define QUANT_TRADING_DNN_ATTENTION_H

#include "layer.h"
#include <vector>
#include <memory>

class Attention : public Layer {
public:
    Attention(size_t input_size, size_t attention_size);
    std::vector<double> forward(const std::vector<double>& input) override;
    std::vector<double> backward(const std::vector<double>& gradOutput) override;
    void updateWeights(double learningRate) override;
    void save(std::ofstream& ofs) const override;
    static std::unique_ptr<Layer> load(std::ifstream& ifs);

public:
    size_t input_size_;
    size_t attention_size_;

    // Weight matrices
    std::vector<std::vector<double>> W_query_;
    std::vector<std::vector<double>> W_key_;
    std::vector<std::vector<double>> W_value_;

    // Gradients
    std::vector<std::vector<double>> grad_W_query_;
    std::vector<std::vector<double>> grad_W_key_;
    std::vector<std::vector<double>> grad_W_value_;

    // Caching for backward pass
    std::vector<double> input_;
    std::vector<double> q_;
    std::vector<double> k_;
    std::vector<double> v_;
    std::vector<double> scores_;
    std::vector<double> weights_;

    // Utility functions
    double softmax(double x, const std::vector<double>& scores, size_t index);
};

#endif //QUANT_TRADING_DNN_ATTENTION_H
