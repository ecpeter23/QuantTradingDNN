//
// Created by Eli Peter on 10/21/24.
//

#ifndef QUANT_TRADING_DNN_BCE_LOSS_H
#define QUANT_TRADING_DNN_BCE_LOSS_H


#include <vector>
#include <cmath>

double computeBCE(const std::vector<double>& predictions, const std::vector<int>& labels) {
  double bce = 0.0;
  for(size_t i = 0; i < predictions.size(); ++i) {
    double p = predictions[i];
    int y = labels[i];
    // To prevent log(0), clamp p
    p = std::min(std::max(p, 1e-15), 1.0 - 1e-15);
    bce += -(y * std::log(p) + (1 - y) * std::log(1 - p));
  }
  return bce / predictions.size();
}

std::vector<double> computeBCEGradient(const std::vector<double>& predictions, const std::vector<int>& labels) {
  std::vector<double> grad(predictions.size(), 0.0);
  for(size_t i = 0; i < predictions.size(); ++i) {
    double p = predictions[i];
    int y = labels[i];
    // To prevent division by zero
    p = std::min(std::max(p, 1e-15), 1.0 - 1e-15);
    grad[i] = (p - y) / (p * (1.0 - p));
  }
  return grad;
}


#endif //QUANT_TRADING_DNN_BCE_LOSS_H
