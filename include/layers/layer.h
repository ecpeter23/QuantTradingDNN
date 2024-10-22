#ifndef QUANT_TRADING_DNN_LAYER_H
#define QUANT_TRADING_DNN_LAYER_H

#include <vector>
#include <memory>
#include <fstream>

class Layer {
public:
    virtual ~Layer() = default;

    virtual std::vector<double> forward(const std::vector<double>& input) = 0;
    virtual std::vector<double> forward(const std::vector<double>& input, bool is_training) = 0;

    virtual std::vector<double> backward(const std::vector<double>& gradOutput) = 0;
    virtual void updateWeights(double learningRate) = 0;

    // Serialization methods
    virtual void save(std::ofstream& ofs) const = 0;
    static std::unique_ptr<Layer> load(std::ifstream& ifs);
};

#endif //QUANT_TRADING_DNN_LAYER_H
