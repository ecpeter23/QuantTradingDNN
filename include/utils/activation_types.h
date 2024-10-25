#ifndef QUANT_TRADING_DNN_ACTIVATION_TYPES_H
#define QUANT_TRADING_DNN_ACTIVATION_TYPES_H


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

#endif //QUANT_TRADING_DNN_ACTIVATION_TYPES_H
