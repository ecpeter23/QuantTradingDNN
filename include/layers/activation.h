#ifndef QUANT_TRADING_DNN_ACTIVATION_H
#define QUANT_TRADING_DNN_ACTIVATION_H

#include "layer.h"
#include "../utils/activation_types.h"
#include "../utils/math.h"
#include <vector>
#include <Accelerate/Accelerate.h>
#include <string>
#include <variant>

class Activation : public Layer {
public:
    explicit Activation(ActivationType type, bool useGPU = false, int axis = -1, int num_linear = 2);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& gradOutput) override;
    void updateWeights(double learningRate) override;

    [[nodiscard]] std::string layerType() const override { return "Activation"; }

private:
    ActivationType type_;
    bool useGPU_;
    int axis_; // Axis along which to apply activations like Softmax and Maxout
    int num_linear_; // Number of linear pieces for Maxout

    Tensor input_; // Store input for backward pass

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
    std::vector<double> maxout_weights_; // Weights for Maxout pieces
    std::vector<double> d_maxout_weights_;

    // Helper functions for CPU-based activations (using Accelerate)
    Tensor applyCPUActivation(const Tensor& input);
    Tensor applyCPUMaxout(const Tensor& input);
    Tensor applyCPUSoftmax(const Tensor& input);

    // Flatten tensor to 1D vector
    static std::vector<double> flattenTensor(const Tensor& tensor);

    // Reconstruct tensor from 1D vector based on original shape
    Tensor reconstructTensor(const std::vector<double>& flat, const Tensor& original);

    // Get total number of elements in tensor
    size_t getTotalElements(const Tensor& tensor);

    // Get shape of tensor
    std::vector<size_t> getShape(const Tensor& tensor);

    // Reshape 1D vector to match tensor shape
    Tensor reshapeTensor(const std::vector<double>& flat, const std::vector<size_t>& shape);
};

#endif //QUANT_TRADING_DNN_ACTIVATION_H
