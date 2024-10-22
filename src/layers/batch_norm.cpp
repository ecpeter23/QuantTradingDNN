#include "layers/batch_norm.h"
#include <cmath>
#include <iostream>

BatchNorm::BatchNorm(size_t num_features, double momentum, double epsilon)
        : num_features_(num_features), momentum_(momentum), epsilon_(epsilon),
          gamma_(num_features, 1.0), beta_(num_features, 0.0),
          running_mean_(num_features, 0.0), running_var_(num_features, 1.0),
          grad_gamma_(num_features, 0.0), grad_beta_(num_features, 0.0) {}

std::vector<double> BatchNorm::forward(const std::vector<double>& input) {
  // Compute mean
  mean_.resize(num_features_);
  for(size_t i = 0; i < num_features_; ++i) {
    mean_[i] = input[i];
  }
  // Assuming input is a single sample
  // Update running mean and variance
  for(size_t i = 0; i < num_features_; ++i) {
    running_mean_[i] = momentum_ * running_mean_[i] + (1.0 - momentum_) * mean_[i];
    running_var_[i] = momentum_ * running_var_[i] + (1.0 - momentum_) * var_[i];
  }

  // Compute variance
  var_.resize(num_features_);
  for(size_t i = 0; i < num_features_; ++i) {
    var_[i] = 0.0; // Since single sample, variance is zero
  }

  // Normalize
  x_hat_.resize(num_features_);
  std_inv_.resize(num_features_);
  std::vector<double> output(num_features_);
  for(size_t i = 0; i < num_features_; ++i) {
    std_inv_[i] = 1.0 / std::sqrt(var_[i] + epsilon_);
    x_hat_[i] = (input[i] - mean_[i]) * std_inv_[i];
    output[i] = gamma_[i] * x_hat_[i] + beta_[i];
  }
  return output;
}

std::vector<double> BatchNorm::backward(const std::vector<double>& gradOutput) {
  // Simplified backward pass for single sample
  std::vector<double> gradInput(num_features_, 0.0);
  for(size_t i = 0; i < num_features_; ++i) {
    grad_gamma_[i] += gradOutput[i] * x_hat_[i];
    grad_beta_[i] += gradOutput[i];
    gradInput[i] = gamma_[i] * gradOutput[i] * std_inv_[i];
  }
  return gradInput;
}

void BatchNorm::updateWeights(double learningRate) {
  for(size_t i = 0; i < num_features_; ++i) {
    gamma_[i] -= learningRate * grad_gamma_[i];
    beta_[i] -= learningRate * grad_beta_[i];
    // Reset gradients after update
    grad_gamma_[i] = 0.0;
    grad_beta_[i] = 0.0;
  }
}

void BatchNorm::save(std::ofstream& ofs) const {
  // Save layer type identifier
  std::string layer_type = "BatchNorm";
  size_t type_length = layer_type.size();
  ofs.write(reinterpret_cast<const char*>(&type_length), sizeof(size_t));
  ofs.write(layer_type.c_str(), type_length);

  // Save parameters
  ofs.write(reinterpret_cast<const char*>(&num_features_), sizeof(size_t));
  ofs.write(reinterpret_cast<const char*>(&momentum_), sizeof(double));
  ofs.write(reinterpret_cast<const char*>(&epsilon_), sizeof(double));

  // Save gamma and beta
  ofs.write(reinterpret_cast<const char*>(gamma_.data()), gamma_.size() * sizeof(double));
  ofs.write(reinterpret_cast<const char*>(beta_.data()), beta_.size() * sizeof(double));

  // Save running_mean and running_var
  ofs.write(reinterpret_cast<const char*>(running_mean_.data()), running_mean_.size() * sizeof(double));
  ofs.write(reinterpret_cast<const char*>(running_var_.data()), running_var_.size() * sizeof(double));
}

std::unique_ptr<Layer> BatchNorm::load(std::ifstream& ifs) {
  size_t num_features;
  double momentum, epsilon;
  ifs.read(reinterpret_cast<char*>(&num_features), sizeof(size_t));
  ifs.read(reinterpret_cast<char*>(&momentum), sizeof(double));
  ifs.read(reinterpret_cast<char*>(&epsilon), sizeof(double));

  auto layer = std::make_unique<BatchNorm>(num_features, momentum, epsilon);

  // Load gamma and beta
  ifs.read(reinterpret_cast<char*>(layer->gamma_.data()), layer->gamma_.size() * sizeof(double));
  ifs.read(reinterpret_cast<char*>(layer->beta_.data()), layer->beta_.size() * sizeof(double));

  // Load running_mean and running_var
  ifs.read(reinterpret_cast<char*>(layer->running_mean_.data()), layer->running_mean_.size() * sizeof(double));
  ifs.read(reinterpret_cast<char*>(layer->running_var_.data()), layer->running_var_.size() * sizeof(double));

  return layer;
}