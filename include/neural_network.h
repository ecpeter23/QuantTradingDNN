#ifndef QUANT_TRADING_DNN_NEURAL_NETWORK_H
#define QUANT_TRADING_DNN_NEURAL_NETWORK_H

#include <vector>
#include <memory>
#include "layers/layer.h"

class NeuralNetwork {
public:
    NeuralNetwork();
    template <typename LayerType, typename... Args>
    void addLayer(Args&&... args);
    void train(const std::vector<std::vector<double>>& data, const std::vector<int>& labels, int epochs, double learningRate);
    double evaluate(const std::vector<std::vector<double>>& data, const std::vector<int>& labels);
    void saveModel(const std::string& filepath);
    void loadModel(const std::string& filepath);

    std::vector<int> predict(const std::vector<std::vector<double>>& data);
protected:
    std::vector<std::unique_ptr<Layer>> layers_;
    // Add any necessary member variables, e.g., loss function, optimizer
};

template <typename LayerType, typename... Args>
void NeuralNetwork::addLayer(Args&&... args) {
  layers_.emplace_back(std::make_unique<LayerType>(std::forward<Args>(args)...));
}

#endif //QUANT_TRADING_DNN_NEURAL_NETWORK_H
