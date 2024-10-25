#ifndef QUANT_TRADING_DNN_ACTIVATIONS_H
#define QUANT_TRADING_DNN_ACTIVATIONS_H

#include "tensor.h"

enum class ActivationType {
    ReLU,             // Rectified Linear Unit
    LeakyReLU,        // Leaky ReLU
    ParametricReLU,   // Parametric ReLU (PReLU)
    ELU,              // Exponential Linear Unit
    SELU,             // Scaled Exponential Linear Unit
    Sigmoid,          // Logistic Sigmoid
    Tanh,             // Hyperbolic Tangent
    HardSigmoid,      // Approximate version of Sigmoid
    HardTanh,         // Clipped version of Tanh
    Softmax,          // Multi-class output activation
    Softplus,         // Smooth version of ReLU
    Softsign,         // Smooth approximation of sign function
    Swish,            // Swish activation (x * sigmoid(x))
    GELU,             // Gaussian Error Linear Unit (Transformers)
    Maxout,           // Maxout for dynamic capacity
    Mish,             // Mish activation function
    ThresholdedReLU,  // ReLU with a threshold
    APL,              // Adaptive Piecewise Linear
    BentIdentity,     // Hybrid activation function
    ESwish,           // Modified Swish with learnable scaling
    LogSigmoid,       // Logarithmic sigmoid for stability
    Sinc,             // Sinc function for signal processing
    TanhExp           // Tanh-Exponential hybrid
};

namespace activations {

    Tensor relu(const Tensor& input);
    Tensor relu_derivative(const Tensor& input);

    Tensor leaky_relu(const Tensor& input, double alpha = 0.01);
    Tensor leaky_relu_derivative(const Tensor& input, double alpha = 0.01);

    Tensor parametric_relu(const Tensor& input, double alpha = 0.01);
    Tensor parametric_relu_derivative(const Tensor& input, double alpha = 0.01);

    Tensor elu(const Tensor& input, double alpha = 1.0);
    Tensor elu_derivative(const Tensor& input, double alpha = 1.0);

    Tensor selu(const Tensor& input);
    Tensor selu_derivative(const Tensor& input);

    Tensor sigmoid(const Tensor& input);
    Tensor sigmoid_derivative(const Tensor& input);

    Tensor tanh(const Tensor& input);
    Tensor tanh_derivative(const Tensor& input);

    Tensor hard_sigmoid(const Tensor& input);
    Tensor hard_sigmoid_derivative(const Tensor& input);

    Tensor hard_tanh(const Tensor& input);
    Tensor hard_tanh_derivative(const Tensor& input);

    Tensor softmax(const Tensor& input);
    Tensor softmax_derivative(const Tensor& input);

    Tensor softplus(const Tensor& input);
    Tensor softplus_derivative(const Tensor& input);

    Tensor softsign(const Tensor& input);
    Tensor softsign_derivative(const Tensor& input);

    Tensor swish(const Tensor& input);
    Tensor swish_derivative(const Tensor& input);

    Tensor gelu(const Tensor& input);
    Tensor gelu_derivative(const Tensor& input);

    Tensor maxout(const Tensor& input, int num_linear = 2);
    Tensor maxout_derivative(const Tensor& input, int num_linear = 2);

    Tensor mish(const Tensor& input);
    Tensor mish_derivative(const Tensor& input);

    Tensor thresholded_relu(const Tensor& input, double theta = 1.0);
    Tensor thresholded_relu_derivative(const Tensor& input, double theta = 1.0);

    Tensor apl(const Tensor& input, const std::vector<double>& params);
    Tensor apl_derivative(const Tensor& input, const std::vector<double>& params);

    Tensor bent_identity(const Tensor& input);
    Tensor bent_identity_derivative(const Tensor& input);

    Tensor eswish(const Tensor& input, double beta = 1.375);
    Tensor eswish_derivative(const Tensor& input, double beta = 1.375);

    Tensor log_sigmoid(const Tensor& input);
    Tensor log_sigmoid_derivative(const Tensor& input);

    Tensor sinc(const Tensor& input);
    Tensor sinc_derivative(const Tensor& input);

    Tensor tanh_exp(const Tensor& input);
    Tensor tanh_exp_derivative(const Tensor& input);
}

#endif //QUANT_TRADING_DNN_ACTIVATIONS_H
