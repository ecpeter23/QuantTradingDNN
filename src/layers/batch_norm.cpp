#include "layers/batch_norm.h"
#include <cmath>
#include <iostream>
#include <Accelerate/Accelerate.h>

// Constructor
BatchNorm::BatchNorm(size_t num_features, double momentum, double epsilon)
        : num_features_(num_features), momentum_(momentum), epsilon_(epsilon),
          gamma_(num_features, 1.0), beta_(num_features, 0.0),
          running_mean_(num_features, 0.0), running_var_(num_features, 1.0),
          grad_gamma_(num_features, 0.0), grad_beta_(num_features, 0.0),
          batch_size_(0), is_training_(true) {}

// Forward pass
std::vector<double> BatchNorm::forward(const std::vector<double>& input) {
  // Assume input is a flattened 2D vector: features x batch_size
  // For example, for 3 features and batch_size 2: [f1s1, f1s2, f2s1, f2s2, f3s1, f3s2]

  // Determine batch size
  if (input.size() % num_features_ != 0) {
    throw std::runtime_error("Input size is not a multiple of num_features_");
  }
  batch_size_ = input.size() / num_features_;

  // Initialize output
  std::vector<double> output(input.size(), 0.0);

  // Compute mean for each feature across the batch using Accelerate
  mean_.resize(num_features_);
  for (size_t i = 0; i < num_features_; ++i) {
    // Pointer to the start of the i-th feature
    const double* feature_ptr = &input[i * batch_size_];
    double feature_mean;
    vDSP_meanvD(feature_ptr, 1, &feature_mean, batch_size_);
    mean_[i] = feature_mean;
  }

  // Compute variance for each feature across the batch using Accelerate
  var_.resize(num_features_);
  for (size_t i = 0; i < num_features_; ++i) {
    const double* feature_ptr = &input[i * batch_size_];
    double feature_var;
    // Compute variance: E[x^2] - (E[x])^2
    double mean_sq;
    double mean_of_sq;
    vDSP_meanvD(feature_ptr, 1, &mean_of_sq, batch_size_);
    std::vector<double> squared(batch_size_);
    vDSP_vsqD(feature_ptr, 1, squared.data(), 1, batch_size_);
    vDSP_meanvD(squared.data(), 1, &mean_sq, batch_size_);
    feature_var = mean_sq - (mean_of_sq * mean_of_sq);
    var_[i] = feature_var;
  }

  if (is_training_) {
    // Update running mean and variance
    for (size_t i = 0; i < num_features_; ++i) {
      running_mean_[i] = momentum_ * running_mean_[i] + (1.0 - momentum_) * mean_[i];
      running_var_[i] = momentum_ * running_var_[i] + (1.0 - momentum_) * var_[i];
    }
  }

  // Decide which mean and variance to use
  std::vector<double> use_mean(num_features_);
  std::vector<double> use_var(num_features_);
  if (is_training_) {
    use_mean = mean_;
    use_var = var_;
  } else {
    use_mean = running_mean_;
    use_var = running_var_;
  }

  // Compute x_hat and output using Accelerate
  x_hat_.resize(input.size(), 0.0);
  std_inv_.resize(num_features_, 0.0);

  for (size_t i = 0; i < num_features_; ++i) {
    // Compute 1 / sqrt(var + epsilon)
    std_inv_[i] = 1.0 / std::sqrt(use_var[i] + epsilon_);
  }

  // Normalize and scale
  for (size_t i = 0; i < num_features_; ++i) {
    const double* feature_ptr = &input[i * batch_size_];
    double gamma = gamma_[i];
    double beta = beta_[i];
    double inv_std = std_inv_[i];

    // Compute x_hat = (x - mean) * inv_std
    std::vector<double> normalized(batch_size_);
    for (size_t j = 0; j < batch_size_; ++j) {
      normalized[j] = (feature_ptr[j] - use_mean[i]) * inv_std;
    }

    // Store x_hat for backward pass
    std::copy(normalized.begin(), normalized.end(), &x_hat_[i * batch_size_]);

    // Compute output = gamma * x_hat + beta
    std::vector<double> scaled(batch_size_);
    vDSP_vsmsaD(normalized.data(), 1, &gamma, &beta, scaled.data(), 1, batch_size_);
    std::copy(scaled.begin(), scaled.end(), &output[i * batch_size_]);
  }

  return output;
}

// Backward pass
std::vector<double> BatchNorm::backward(const std::vector<double>& gradOutput) {
  // gradOutput is a flattened 2D vector: features x batch_size
  if (gradOutput.size() != num_features_ * batch_size_) {
    throw std::runtime_error("gradOutput size mismatch");
  }

  std::vector<double> gradInput(gradOutput.size(), 0.0);

  // Compute gradients w.r. to gamma and beta
  for (size_t i = 0; i < num_features_; ++i) {
    const double* grad_ptr = &gradOutput[i * batch_size_];
    const double* x_hat_ptr = &x_hat_[i * batch_size_];

    // grad_gamma += sum over batch (gradOutput * x_hat)
    double grad_gamma = 0.0;
    vDSP_dotprD(grad_ptr, 1, x_hat_ptr, 1, &grad_gamma, batch_size_);
    grad_gamma_[i] += grad_gamma;

    // grad_beta += sum over batch (gradOutput)
    double grad_beta = 0.0;
    vDSP_sveD(grad_ptr, 1, &grad_beta, batch_size_);
    grad_beta_[i] += grad_beta;

    // Compute gradInput
    double gamma = gamma_[i];
    double inv_std = std_inv_[i];

    // gradInput = gamma * gradOutput * inv_std
    std::vector<double> temp(batch_size_);
    vDSP_vsmsaD(grad_ptr, 1, &gamma, &temp[0], 1, batch_size_);
    vDSP_vsmulD(temp.data(), 1, &inv_std, temp.data(), 1, batch_size_);
    std::copy(temp.begin(), temp.end(), &gradInput[i * batch_size_]);
  }

  // For a more accurate backward pass, especially when computing gradients w.r.t input,
  // you should consider the full gradient computation involving mean and variance gradients.
  // However, for simplicity, this implementation provides a basic gradient computation.

  return gradInput;
}

// Update weights
void BatchNorm::updateWeights(double learningRate) {
  for (size_t i = 0; i < num_features_; ++i) {
    gamma_[i] -= learningRate * grad_gamma_[i];
    beta_[i] -= learningRate * grad_beta_[i];

    // Reset gradients after update
    grad_gamma_[i] = 0.0;
    grad_beta_[i] = 0.0;
  }
}

// Save layer parameters
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

// Load layer parameters
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
