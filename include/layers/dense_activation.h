#ifndef QUANT_TRADING_DNN_DENSE_ACTIVATION_H
#define QUANT_TRADING_DNN_DENSE_ACTIVATION_H

#include "layer.h"
#include "utils/types.h"
#include <vector>
#include <memory>


class DenseActivation : public Layer {
public:
    DenseActivation(size_t input_size, size_t output_size, ActivationType activation);
    std::vector<double> forward(const std::vector<double>& input) override;
    std::vector<double> backward(const std::vector<double>& gradOutput) override;
    void updateWeights(double learningRate) override;
    void save(std::ofstream& ofs) const override;
    static std::unique_ptr<Layer> load(std::ifstream& ifs);

public:
    size_t input_size_;
    size_t output_size_;
    ActivationType activation_;

    std::vector<std::vector<double>> weights_; // [output_size_][input_size_]
    std::vector<double> biases_;               // [output_size_]

    // Gradients
    std::vector<std::vector<double>> grad_weights_; // [output_size_][input_size_]
    std::vector<double> grad_biases_;               // [output_size_]

    // Caching for backward pass
    std::vector<double> input_;    // Cached input vector
    std::vector<double> linear_;   // Cached linear transformation outputs

    // Activation functions
    double activate(double x) const;
    double activateDerivative(double x) const;
};

#endif //QUANT_TRADING_DNN_DENSE_ACTIVATION_H
