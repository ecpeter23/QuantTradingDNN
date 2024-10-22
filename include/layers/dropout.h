#ifndef QUANT_TRADING_DNN_DROPOUT_H
#define QUANT_TRADING_DNN_DROPOUT_H


#include "layer.h"
#include <vector>
#include <random>

class Dropout : public Layer {
public:
    explicit Dropout(double rate);
    std::vector<double> forward(const std::vector<double>& input) override;
    std::vector<double> backward(const std::vector<double>& gradOutput) override;
    void updateWeights(double learningRate) override {}
    void save(std::ofstream& ofs) const override;
    static std::unique_ptr<Layer> load(std::ifstream& ifs);

public:
    double rate_;
    std::vector<bool> mask_;
    std::mt19937 gen_;
    std::bernoulli_distribution dist_;
};


#endif //QUANT_TRADING_DNN_DROPOUT_H
