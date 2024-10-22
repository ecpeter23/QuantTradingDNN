#ifndef QUANT_TRADING_DNN_FULLY_CONNECTED_H
#define QUANT_TRADING_DNN_FULLY_CONNECTED_H

#include "layer.h"
#include <vector>

class FullyConnected : public Layer {
public:
    FullyConnected(int inputSize, int outputSize);
    std::vector<double> forward(const std::vector<double>& input) override;
    std::vector<double> backward(const std::vector<double>& gradOutput) override;
    void updateWeights(double learningRate) override;

    void save(std::ofstream &ofs) const override;
    static std::unique_ptr<Layer> load(std::ifstream &ifs);

public:
    int inputSize_;
    int outputSize_;
    std::vector<std::vector<double>> weights_;
    std::vector<double> biases_;
    std::vector<double> input_;
    // Gradients
    std::vector<std::vector<double>> gradWeights_;
    std::vector<double> gradBiases_;
};

#endif //QUANT_TRADING_DNN_FULLY_CONNECTED_H
