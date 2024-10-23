#ifndef QUANT_TRADING_DNN_ACTIVATION_H
#define QUANT_TRADING_DNN_ACTIVATION_H

#include "layer.h"
#include "../utils/types.h"
#include <vector>
#include <Accelerate/Accelerate.h>
#include <string>

class Activation : public Layer {
public:
    explicit Activation(ActivationType type, bool useGPU = false);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& gradOutput) override;
    void updateWeights(double learningRate) override;

    [[nodiscard]] std::string layerType() const override { return "Activation"; }

private:
    ActivationType type_;
    bool useGPU_;
    std::vector<double> input_;

    // Parameters for specific activations
    // ParametricReLU
    double alpha_;
    double d_alpha_;

    // ThresholdedReLU
    double theta_;
    double d_theta_;

    // ESwish
    double beta_;
    double d_beta_;

    // APL
    std::vector<double> apl_params_;
    std::vector<double> d_apl_params_;

    // Maxout
    int num_linear_;
    std::vector<double> maxout_weights_; // Simplified representation
    std::vector<double> d_maxout_weights_;

    // Helper functions for CPU-based activations (using vDSP)
    Tensor applyCPUActivation(const std::vector<double>& input);
    Tensor applyGPUActivation(const std::vector<double>& input);
};

#endif //QUANT_TRADING_DNN_ACTIVATION_H
