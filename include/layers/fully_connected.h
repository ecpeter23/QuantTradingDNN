#ifndef QUANT_TRADING_DNN_FULLY_CONNECTED_H
#define QUANT_TRADING_DNN_FULLY_CONNECTED_H

#include "layer.h"
#include <vector>

class FullyConnected : public Layer {
public:
    FullyConnected(int inputSize, int outputSize);

    // Forward and Backward passes
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& gradOutput) override;

    // Weight update method using gradients
    void updateWeights(double learningRate) override;

    // Save and Load layer parameters for serialization
    void save(std::ofstream &ofs) const override;
    static std::unique_ptr<Layer> load(std::ifstream &ifs);

    [[nodiscard]] std::string layerType() const override { return "FullyConnected"; }

private:
    int inputSize_;
    int outputSize_;

    // Layer parameters
    std::vector<std::vector<double>> weights_;
    std::vector<double> biases_;

    // Inputs stored for backpropagation
    std::vector<double> input_;

    // Gradients for weights and biases
    std::vector<std::vector<double>> gradWeights_;
    std::vector<double> gradBiases_;
};

#endif //QUANT_TRADING_DNN_FULLY_CONNECTED_H
