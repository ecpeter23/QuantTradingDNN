//
// Created by Eli Peter on 10/21/24.
//

#ifndef QUANT_TRADING_DNN_MATH_H
#define QUANT_TRADING_DNN_MATH_H

#include <vector>
#include <cmath>

// Matrix-vector multiplication using Accelerate framework
std::vector<double> matMul(const std::vector<std::vector<double>>& matrix, const std::vector<double>& vec);

// ACTIVATION FUNCTIONS AND DERIVATIVES
inline double relu_derivative(double x) {
  return x > 0 ? 1.0 : 0.0;
}

inline double leaky_relu_derivative(double x, double alpha = 0.01) {
  return x > 0 ? 1.0 : alpha;
}

inline double sigmoid(double x) {
  return 1.0 / (1.0 + std::exp(-x));
}

inline double sigmoid_derivative(double x) {
  double s = sigmoid(x);
  return s * (1.0 - s);
}

inline double tanh_derivative(double x) {
  double t = std::tanh(x);
  return 1.0 - t * t;
}

inline double elu_derivative(double x, double alpha = 1.0) {
  return x > 0 ? 1.0 : alpha * std::exp(x);
}

inline double softplus(double x) {
  return std::log(1.0 + std::exp(x));
}

inline double softplus_derivative(double x) {
  return sigmoid(x);
}

inline double softsign(double x) {
  return x / (1.0 + std::abs(x));
}

inline double softsign_derivative(double x) {
  return 1.0 / std::pow(1.0 + std::abs(x), 2);
}

inline double swish(double x) {
  return x * sigmoid(x);
}

inline double swish_derivative(double x) {
  double s = sigmoid(x);
  return s + x * s * (1.0 - s);
}

inline double gelu(double x) {
  return 0.5 * x * (1.0 + std::tanh(std::sqrt(2.0 / M_PI) * (x + 0.044715 * std::pow(x, 3))));
}

inline double gelu_derivative(double x) {
  double t = std::tanh(std::sqrt(2.0 / M_PI) * (x + 0.044715 * std::pow(x, 3)));
  return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * (std::sqrt(2.0 / M_PI) * (1.0 + 3.0 * 0.044715 * x * x));
}

inline double mish(double x) {
  return x * std::tanh(std::log(1.0 + std::exp(x)));
}

inline double mish_derivative(double x) {
  double e_x = std::exp(x);
  double tanh_arg = std::log(1.0 + e_x);
  double tanh_val = std::tanh(tanh_arg);
  double sech2 = 1.0 - tanh_val * tanh_val;
  return tanh_val + x * sech2 * (e_x / (1.0 + e_x));
}

inline double thresholded_relu(double x, double theta = 1.0) {
  return x > theta ? x : 0.0;
}

inline double thresholded_relu_derivative(double x, double theta = 1.0) {
  return x > theta ? 1.0 : 0.0;
}

inline double hard_sigmoid(double x) {
  return std::max(0.0, std::min(1.0, 0.2 * x + 0.5));
}

inline double hard_sigmoid_derivative(double x) {
  if (x < -2.5 || x > 2.5)
    return 0.0;
  else
    return 0.2;
}

inline double hard_tanh(double x) {
  return std::max(-1.0, std::min(1.0, x));
}

inline double hard_tanh_derivative(double x) {
  return (x >= -1.0 && x <= 1.0) ? 1.0 : 0.0;
}

#endif //QUANT_TRADING_DNN_MATH_H
