#include "layers/attention.h"
#include <cmath>
#include <algorithm>
#include <random>

// Implement softmax as a utility function
double Attention::softmax(double x, const std::vector<double>& scores, size_t index) {
  double exp_x = std::exp(x);
  double sum_exp = 0.0;
  for(const auto& score : scores) {
    sum_exp += std::exp(score);
  }
  return exp_x / sum_exp;
}

Attention::Attention(size_t input_size, size_t attention_size)
        : input_size_(input_size), attention_size_(attention_size),
          W_query_(attention_size, std::vector<double>(input_size, 0.0)),
          W_key_(attention_size, std::vector<double>(input_size, 0.0)),
          W_value_(attention_size, std::vector<double>(input_size, 0.0)),
          grad_W_query_(attention_size, std::vector<double>(input_size, 0.0)),
          grad_W_key_(attention_size, std::vector<double>(input_size, 0.0)),
          grad_W_value_(attention_size, std::vector<double>(input_size, 0.0)) {
  // Initialize W_query, W_key, W_value with small random values
  std::mt19937 gen(std::random_device{}());
  std::uniform_real_distribution<> dist(-0.1, 0.1);

  for(auto& row : W_query_) {
    for(auto& w : row) {
      w = dist(gen);
    }
  }
  for(auto& row : W_key_) {
    for(auto& w : row) {
      w = dist(gen);
    }
  }
  for(auto& row : W_value_) {
    for(auto& w : row) {
      w = dist(gen);
    }
  }
}

std::vector<double> Attention::forward(const std::vector<double>& input) {
  input_ = input; // Cache input for backward pass

  // Compute queries, keys, and values
  q_.resize(attention_size_, 0.0);
  k_.resize(attention_size_, 0.0);
  v_.resize(attention_size_, 0.0);

  for(size_t i = 0; i < attention_size_; ++i) {
    for(size_t j = 0; j < input_size_; ++j) {
      q_[i] += W_query_[i][j] * input[j];
      k_[i] += W_key_[i][j] * input[j];
      v_[i] += W_value_[i][j] * input[j];
    }
  }

  // Compute attention scores
  scores_.resize(attention_size_, 0.0);
  double scale = std::sqrt(static_cast<double>(input_size_));
  for(size_t i = 0; i < attention_size_; ++i) {
    scores_[i] = (q_[i] * k_[i]) / scale;
  }

  // Apply softmax to get attention weights
  weights_.resize(attention_size_, 0.0);
  for(size_t i = 0; i < attention_size_; ++i) {
    weights_[i] = softmax(scores_[i], scores_, i);
  }

  // Compute weighted sum of values
  std::vector<double> output(attention_size_, 0.0);
  for(size_t i = 0; i < attention_size_; ++i) {
    output[i] = weights_[i] * v_[i];
  }

  return output;
}

std::vector<double> Attention::backward(const std::vector<double>& gradOutput) {
  // Initialize gradients
  std::fill(grad_W_query_.begin(), grad_W_query_.end(),
            std::vector<double>(input_size_, 0.0));
  std::fill(grad_W_key_.begin(), grad_W_key_.end(),
            std::vector<double>(input_size_, 0.0));
  std::fill(grad_W_value_.begin(), grad_W_value_.end(),
            std::vector<double>(input_size_, 0.0));

  // Compute gradients w.r.t output (attention weights * values)
  // gradOutput is dL/doutput, where output = sum(weights[i] * v[i])
  // Thus, dL/dweights[i] = dL/doutput * v[i]
  //       dL/dv[i] = dL/doutput * weights[i]
  std::vector<double> grad_weights(attention_size_, 0.0);
  std::vector<double> grad_v(attention_size_, 0.0);
  for(size_t i = 0; i < attention_size_; ++i) {
    grad_weights[i] = gradOutput[i] * v_[i];
    grad_v[i] = gradOutput[i] * weights_[i];
  }

  // Compute gradients w.r.t scores using softmax derivative
  // dL/dscores[i] = sum_j dL/dweights[j] * dweights[j]/dscores[i]
  // where dweights[j]/dscores[i] = weights[j] * (delta_ij - weights[i])
  std::vector<double> grad_scores(attention_size_, 0.0);
  for(size_t i = 0; i < attention_size_; ++i) {
    for(size_t j = 0; j < attention_size_; ++j) {
      if(i == j) {
        grad_scores[i] += grad_weights[j] * weights_[j] * (1.0 - weights_[i]);
      } else {
        grad_scores[i] += grad_weights[j] * weights_[j] * (-weights_[i]);
      }
    }
  }

  // Compute gradients w.r.t q and k
  // scores[i] = (q[i] * k[i]) / scale
  // Thus, dL/dq[i] = dL/dscores[i] * k[i] / scale
  //       dL/dk[i] = dL/dscores[i] * q[i] / scale
  std::vector<double> grad_q(attention_size_, 0.0);
  std::vector<double> grad_k(attention_size_, 0.0);
  double scale = std::sqrt(static_cast<double>(input_size_));
  for(size_t i = 0; i < attention_size_; ++i) {
    grad_q[i] = grad_scores[i] * k_[i] / scale;
    grad_k[i] = grad_scores[i] * q_[i] / scale;
  }

  // Compute gradients w.r.t W_query and W_key
  // q[i] = W_query[i][j] * input[j]
  // Thus, dL/dW_query[i][j] += grad_q[i] * input[j]
  for(size_t i = 0; i < attention_size_; ++i) {
    for(size_t j = 0; j < input_size_; ++j) {
      grad_W_query_[i][j] += grad_q[i] * input_[j];
      grad_W_key_[i][j] += grad_k[i] * input_[j];
      grad_W_value_[i][j] += grad_v[i] * input_[j];
    }
  }

  // Compute gradients w.r.t input
  // dL/dinput[j] = sum_i (W_query[i][j] * grad_q[i] + W_key[i][j] * grad_k[i] + W_value[i][j] * grad_v[i])
  std::vector<double> grad_input(input_size_, 0.0);
  for(size_t j = 0; j < input_size_; ++j) {
    for(size_t i = 0; i < attention_size_; ++i) {
      grad_input[j] += W_query_[i][j] * grad_q[i];
      grad_input[j] += W_key_[i][j] * grad_k[i];
      grad_input[j] += W_value_[i][j] * grad_v[i];
    }
  }

  // Store gradients for weight updates
  // (Already stored in grad_W_query_, grad_W_key_, grad_W_value_)

  return grad_input;
}

void Attention::updateWeights(double learningRate) {
  // Update W_query
  for(size_t i = 0; i < attention_size_; ++i) {
    for(size_t j = 0; j < input_size_; ++j) {
      W_query_[i][j] -= learningRate * grad_W_query_[i][j];
      grad_W_query_[i][j] = 0.0; // Reset gradient after update
    }
  }

  // Update W_key
  for(size_t i = 0; i < attention_size_; ++i) {
    for(size_t j = 0; j < input_size_; ++j) {
      W_key_[i][j] -= learningRate * grad_W_key_[i][j];
      grad_W_key_[i][j] = 0.0; // Reset gradient after update
    }
  }

  // Update W_value
  for(size_t i = 0; i < attention_size_; ++i) {
    for(size_t j = 0; j < input_size_; ++j) {
      W_value_[i][j] -= learningRate * grad_W_value_[i][j];
      grad_W_value_[i][j] = 0.0; // Reset gradient after update
    }
  }
}

void Attention::save(std::ofstream& ofs) const {
  // Save layer type identifier
  std::string layer_type = "Attention";
  size_t type_length = layer_type.size();
  ofs.write(reinterpret_cast<const char*>(&type_length), sizeof(size_t));
  ofs.write(layer_type.c_str(), type_length);

  // Save parameters
  ofs.write(reinterpret_cast<const char*>(&input_size_), sizeof(size_t));
  ofs.write(reinterpret_cast<const char*>(&attention_size_), sizeof(size_t));

  // Save W_query, W_key, W_value
  for(const auto& row : W_query_) {
    ofs.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));
  }
  for(const auto& row : W_key_) {
    ofs.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));
  }
  for(const auto& row : W_value_) {
    ofs.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));
  }
}

std::unique_ptr<Layer> Attention::load(std::ifstream& ifs) {
  size_t input_size, attention_size;
  ifs.read(reinterpret_cast<char*>(&input_size), sizeof(size_t));
  ifs.read(reinterpret_cast<char*>(&attention_size), sizeof(size_t));

  auto layer = std::make_unique<Attention>(input_size, attention_size);

  // Load W_query, W_key, W_value
  for(auto& row : layer->W_query_) {
    ifs.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
  }
  for(auto& row : layer->W_key_) {
    ifs.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
  }
  for(auto& row : layer->W_value_) {
    ifs.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
  }

  return layer;
}
