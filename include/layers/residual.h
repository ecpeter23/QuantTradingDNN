//
// Created by Eli Peter on 10/22/24.
//

#ifndef QUANT_TRADING_DNN_RESIDUAL_H
#define QUANT_TRADING_DNN_RESIDUAL_H

#include "layer.h"
#include <vector>
#include <memory>

class Residual : public Layer {
public:
    Residual(std::shared_ptr<Layer> subLayer);
    std::vector<double> forward(const std::vector<double>& input) override;
    std::vector<double> backward(const std::vector<double>& gradOutput) override;
    void updateWeights(double learningRate) override;
    void save(std::ofstream& ofs) const override;
    static std::unique_ptr<Layer> load(std::ifstream& ifs);

public:
    std::shared_ptr<Layer> subLayer_;
};


#endif //QUANT_TRADING_DNN_RESIDUAL_H
