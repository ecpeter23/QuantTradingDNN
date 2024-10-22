#ifndef QUANT_TRADING_DNN_ACTIVATION_H
#define QUANT_TRADING_DNN_ACTIVATION_H

#include "layer.h"
#include <vector>
#include "utils/types.h"

class Activation : public Layer {
public:
    explicit Activation(ActivationType type);
    std::vector<double> forward(const std::vector<double>& input) override;
    std::vector<double> backward(const std::vector<double>& gradOutput) override;
    void updateWeights(double learningRate) override {}

    void save(std::ofstream &ofs) const override;
    static std::unique_ptr<Layer> load(std::ifstream &ifs);

private:
    ActivationType type_;
    std::vector<double> input_;
};

#endif //QUANT_TRADING_DNN_ACTIVATION_H
