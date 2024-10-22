//
// Created by Eli Peter on 10/22/24.
//

#ifndef QUANT_TRADING_DNN_DENSE_ACTIVATION_H
#define QUANT_TRADING_DNN_DENSE_ACTIVATION_H

#include "layer.h"
#include <vector>
#include <functional>

enum class ActivationType {
    ReLU,
    Sigmoid,
    Tanh
};

class DenseActivation : public Layer {
public:
    DenseActivation(size_t input_size, size_t output_size, ActivationType activation);
    std::vector<double> forward(const std::vector<double>& input) override;
    std::vector<double> backward(const std::vector<double>& gradOutput) override;
    void updateWeights(double learningRate) override;
    void save(std::ofstream& ofs) const override;
    static std::unique_ptr<Layer> load(std::ifstream& ifs);

private:
    size_t input_size_;
    size_t output_size_;
    ActivationType activation_;

    std::vector<std::vector<double>> weights_;
    std::vector<double> biases_;

    // Gradients
    // ...

    // Activation functions
    double activate(double x) const;
    double activateDerivative(double x) const;
};


#endif //QUANT_TRADING_DNN_DENSE_ACTIVATION_H
