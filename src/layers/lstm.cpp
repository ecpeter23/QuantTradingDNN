#include "layers/lstm.h"
#include <cmath>
#include <random>
#include <iostream>
#include <Accelerate/Accelerate.h>

// Activation functions
double LSTM::sigmoid(double x) const {
  return 1.0 / (1.0 + std::exp(-x));
}

double LSTM::tanh_func(double x) const {
  return std::tanh(x);
}

// Constructor: Initializes weights, biases, and gradients
LSTM::LSTM(size_t input_size, size_t hidden_size)
        : input_size_(input_size), hidden_size_(hidden_size),
          c_(hidden_size, 0.0), h_(hidden_size, 0.0),
          c_prev_(hidden_size, 0.0), h_prev_(hidden_size, 0.0),
          i_(hidden_size, 0.0), f_(hidden_size, 0.0),
          g_(hidden_size, 0.0), o_(hidden_size, 0.0) {
  // Initialize weight matrices and biases with small random values
  std::mt19937 gen(std::random_device{}());
  std::uniform_real_distribution<> dist(-0.1, 0.1);

  // Initialize W_i, U_i, b_i
  W_i_.resize(hidden_size_, std::vector<double>(input_size_, 0.0));
  U_i_.resize(hidden_size_, std::vector<double>(hidden_size_, 0.0));
  b_i_.resize(hidden_size_, 0.0);
  for(auto& row : W_i_) {
    for(auto& w : row) {
      w = dist(gen);
    }
  }
  for(auto& row : U_i_) {
    for(auto& w : row) {
      w = dist(gen);
    }
  }
  for(auto& b : b_i_) {
    b = dist(gen);
  }

  // Initialize W_f, U_f, b_f
  W_f_.resize(hidden_size_, std::vector<double>(input_size_, 0.0));
  U_f_.resize(hidden_size_, std::vector<double>(hidden_size_, 0.0));
  b_f_.resize(hidden_size_, 0.0);
  for(auto& row : W_f_) {
    for(auto& w : row) {
      w = dist(gen);
    }
  }
  for(auto& row : U_f_) {
    for(auto& w : row) {
      w = dist(gen);
    }
  }
  for(auto& b : b_f_) {
    b = dist(gen);
  }

  // Initialize W_c, U_c, b_c
  W_c_.resize(hidden_size_, std::vector<double>(input_size_, 0.0));
  U_c_.resize(hidden_size_, std::vector<double>(hidden_size_, 0.0));
  b_c_.resize(hidden_size_, 0.0);
  for(auto& row : W_c_) {
    for(auto& w : row) {
      w = dist(gen);
    }
  }
  for(auto& row : U_c_) {
    for(auto& w : row) {
      w = dist(gen);
    }
  }
  for(auto& b : b_c_) {
    b = dist(gen);
  }

  // Initialize W_o, U_o, b_o
  W_o_.resize(hidden_size_, std::vector<double>(input_size_, 0.0));
  U_o_.resize(hidden_size_, std::vector<double>(hidden_size_, 0.0));
  b_o_.resize(hidden_size_, 0.0);
  for(auto& row : W_o_) {
    for(auto& w : row) {
      w = dist(gen);
    }
  }
  for(auto& row : U_o_) {
    for(auto& w : row) {
      w = dist(gen);
    }
  }
  for(auto& b : b_o_) {
    b = dist(gen);
  }

  // Initialize gradients to zero
  grad_W_i_.resize(hidden_size_, std::vector<double>(input_size_, 0.0));
  grad_U_i_.resize(hidden_size_, std::vector<double>(hidden_size_, 0.0));
  grad_b_i_.resize(hidden_size_, 0.0);

  grad_W_f_.resize(hidden_size_, std::vector<double>(input_size_, 0.0));
  grad_U_f_.resize(hidden_size_, std::vector<double>(hidden_size_, 0.0));
  grad_b_f_.resize(hidden_size_, 0.0);

  grad_W_c_.resize(hidden_size_, std::vector<double>(input_size_, 0.0));
  grad_U_c_.resize(hidden_size_, std::vector<double>(hidden_size_, 0.0));
  grad_b_c_.resize(hidden_size_, 0.0);

  grad_W_o_.resize(hidden_size_, std::vector<double>(input_size_, 0.0));
  grad_U_o_.resize(hidden_size_, std::vector<double>(hidden_size_, 0.0));
  grad_b_o_.resize(hidden_size_, 0.0);
}

// Forward pass: Computes gates and updates cell and hidden states
std::vector<double> LSTM::forward(const std::vector<double>& input) {
  // Cache the input for use in backward pass
  input_ = input;

  // Cache previous states
  c_prev_ = c_;
  h_prev_ = h_;

  // Compute input gate: i = sigmoid(W_i * input + U_i * h_prev + b_i)
  // Matrix multiplication using Accelerate framework
  std::vector<double> Wx_i(hidden_size_, 0.0);
  // Using vDSP_mmulD for matrix-vector multiplication
  // Matrix A (W_i_) is [hidden_size_][input_size_], vector x is [input_size_]
  // Result y = A * x is [hidden_size_]
  vDSP_mmulD(W_i_[0].data(), 1, input.data(), 1, Wx_i.data(), 1, hidden_size_, 1, input_size_);

  std::vector<double> Uh_i(hidden_size_, 0.0);
  vDSP_mmulD(U_i_[0].data(), 1, h_prev_.data(), 1, Uh_i.data(), 1, hidden_size_, 1, hidden_size_);

  // Sum Wx_i + Uh_i + b_i_
  std::vector<double> gate_i(hidden_size_, 0.0);
  for(size_t j = 0; j < hidden_size_; ++j) {
    gate_i[j] = Wx_i[j] + Uh_i[j] + b_i_[j];
    i_[j] = sigmoid(gate_i[j]);
  }

  // Compute forget gate: f = sigmoid(W_f * input + U_f * h_prev + b_f)
  std::vector<double> Wx_f(hidden_size_, 0.0);
  vDSP_mmulD(W_f_[0].data(), 1, input.data(), 1, Wx_f.data(), 1, hidden_size_, 1, input_size_);

  std::vector<double> Uh_f(hidden_size_, 0.0);
  vDSP_mmulD(U_f_[0].data(), 1, h_prev_.data(), 1, Uh_f.data(), 1, hidden_size_, 1, hidden_size_);

  // Sum Wx_f + Uh_f + b_f_
  std::vector<double> gate_f(hidden_size_, 0.0);
  for(size_t j = 0; j < hidden_size_; ++j) {
    gate_f[j] = Wx_f[j] + Uh_f[j] + b_f_[j];
    f_[j] = sigmoid(gate_f[j]);
  }

  // Compute cell gate: g = tanh(W_c * input + U_c * h_prev + b_c)
  std::vector<double> Wx_c(hidden_size_, 0.0);
  vDSP_mmulD(W_c_[0].data(), 1, input.data(), 1, Wx_c.data(), 1, hidden_size_, 1, input_size_);

  std::vector<double> Uh_c(hidden_size_, 0.0);
  vDSP_mmulD(U_c_[0].data(), 1, h_prev_.data(), 1, Uh_c.data(), 1, hidden_size_, 1, hidden_size_);

  // Sum Wx_c + Uh_c + b_c_
  std::vector<double> gate_c(hidden_size_, 0.0);
  for(size_t j = 0; j < hidden_size_; ++j) {
    gate_c[j] = Wx_c[j] + Uh_c[j] + b_c_[j];
    g_[j] = tanh_func(gate_c[j]);
  }

  // Update cell state: c = f * c_prev + i * g
  for(size_t j = 0; j < hidden_size_; ++j) {
    c_[j] = f_[j] * c_prev_[j] + i_[j] * g_[j];
  }

  // Compute output gate: o = sigmoid(W_o * input + U_o * h_prev + b_o)
  std::vector<double> Wx_o(hidden_size_, 0.0);
  vDSP_mmulD(W_o_[0].data(), 1, input.data(), 1, Wx_o.data(), 1, hidden_size_, 1, input_size_);

  std::vector<double> Uh_o(hidden_size_, 0.0);
  vDSP_mmulD(U_o_[0].data(), 1, h_prev_.data(), 1, Uh_o.data(), 1, hidden_size_, 1, hidden_size_);

  // Sum Wx_o + Uh_o + b_o_
  std::vector<double> gate_o(hidden_size_, 0.0);
  for(size_t j = 0; j < hidden_size_; ++j) {
    gate_o[j] = Wx_o[j] + Uh_o[j] + b_o_[j];
    o_[j] = sigmoid(gate_o[j]);
  }

  // Update hidden state: h = o * tanh(c)
  for(size_t j = 0; j < hidden_size_; ++j) {
    h_[j] = o_[j] * tanh_func(c_[j]);
  }

  return h_;
}

// Backward pass: Computes gradients with respect to inputs, weights, and biases
std::vector<double> LSTM::backward(const std::vector<double>& gradOutput) {
  // Initialize gradients w.r.t h and c
  std::vector<double> grad_h = gradOutput; // dL/dh_t
  std::vector<double> grad_c(hidden_size_, 0.0); // dL/dc_t

  // Compute dL/dc_t = dL/dh_t * o * (1 - tanh(c)^2)
  std::vector<double> tanh_c(hidden_size_, 0.0);
  for(size_t j = 0; j < hidden_size_; ++j) {
    tanh_c[j] = tanh_func(c_[j]);
  }

  for(size_t j = 0; j < hidden_size_; ++j) {
    grad_c[j] = grad_h[j] * o_[j] * (1.0 - tanh_c[j] * tanh_c[j]);
  }

  // Compute gradients w.r.t output gate: dL/do = dL/dh * tanh(c)
  std::vector<double> grad_o(hidden_size_, 0.0);
  for(size_t j = 0; j < hidden_size_; ++j) {
    grad_o[j] = grad_h[j] * tanh_c[j];
  }

  // Compute gradients w.r.t cell gate: dL/dg = dL/dc * i
  std::vector<double> grad_g(hidden_size_, 0.0);
  for(size_t j = 0; j < hidden_size_; ++j) {
    grad_g[j] = grad_c[j] * i_[j];
  }

  // Compute gradients w.r.t input gate: dL/di = dL/dc * g
  std::vector<double> grad_i(hidden_size_, 0.0);
  for(size_t j = 0; j < hidden_size_; ++j) {
    grad_i[j] = grad_c[j] * g_[j];
  }

  // Compute gradients w.r.t forget gate: dL/df = dL/dc * c_prev
  std::vector<double> grad_f(hidden_size_, 0.0);
  for(size_t j = 0; j < hidden_size_; ++j) {
    grad_f[j] = grad_c[j] * c_prev_[j];
  }

  // Compute derivatives of gate activations
  // dL/dgate = dL/dgate_act * activation_derivative(gate)
  std::vector<double> dL_dgate_o(hidden_size_, 0.0);
  std::vector<double> dL_dgate_g(hidden_size_, 0.0);
  std::vector<double> dL_dgate_i(hidden_size_, 0.0);
  std::vector<double> dL_dgate_f(hidden_size_, 0.0);

  for(size_t j = 0; j < hidden_size_; ++j) {
    // Output gate derivative
    dL_dgate_o[j] = grad_o[j] * o_[j] * (1.0 - o_[j]);

    // Cell gate derivative
    dL_dgate_g[j] = grad_g[j] * (1.0 - g_[j] * g_[j]);

    // Input gate derivative
    dL_dgate_i[j] = grad_i[j] * i_[j] * (1.0 - i_[j]);

    // Forget gate derivative
    dL_dgate_f[j] = grad_f[j] * f_[j] * (1.0 - f_[j]);
  }

  // Compute gradients for each gate
  // Initialize gradients w.r.t inputs and h_prev
  std::vector<double> grad_input(input_size_, 0.0);
  std::vector<double> grad_h_prev(hidden_size_, 0.0);

  // Define a lambda to compute gradients for a gate
  auto compute_gate_gradients = [&](const std::vector<double>& dL_dgate,
                                    const std::vector<std::vector<double>>& W,
                                    const std::vector<std::vector<double>>& U,
                                    std::vector<double>& grad_input,
                                    std::vector<double>& grad_h_prev,
                                    std::vector<std::vector<double>>& grad_W,
                                    std::vector<std::vector<double>>& grad_U,
                                    std::vector<double>& grad_b) {
      // Compute grad_W += dL_dgate * input_^T
      for(size_t i = 0; i < hidden_size_; ++i) {
        for(size_t j = 0; j < input_size_; ++j) {
          grad_W[i][j] += dL_dgate[i] * input_[j];
        }
      }

      // Compute grad_U += dL_dgate * h_prev_^T
      for(size_t i = 0; i < hidden_size_; ++i) {
        for(size_t j = 0; j < hidden_size_; ++j) {
          grad_U[i][j] += dL_dgate[i] * h_prev_[j];
        }
      }

      // Compute grad_b += dL_dgate
      for(size_t i = 0; i < hidden_size_; ++i) {
        grad_b[i] += dL_dgate[i];
      }

      // Compute grad_input += W^T * dL_dgate
      for(size_t j = 0; j < input_size_; ++j) {
        for(size_t i = 0; i < hidden_size_; ++i) {
          grad_input[j] += W[i][j] * dL_dgate[i];
        }
      }

      // Compute grad_h_prev += U^T * dL_dgate
      for(size_t j = 0; j < hidden_size_; ++j) {
        for(size_t i = 0; i < hidden_size_; ++i) {
          grad_h_prev[j] += U[i][j] * dL_dgate[i];
        }
      }
  };

  // Compute gradients for output gate
  compute_gate_gradients(dL_dgate_o, W_o_, U_o_,
                         grad_input, grad_h_prev,
                         grad_W_o_, grad_U_o_, grad_b_o_);

  // Compute gradients for cell gate
  compute_gate_gradients(dL_dgate_g, W_c_, U_c_,
                         grad_input, grad_h_prev,
                         grad_W_c_, grad_U_c_, grad_b_c_);

  // Compute gradients for input gate
  compute_gate_gradients(dL_dgate_i, W_i_, U_i_,
                         grad_input, grad_h_prev,
                         grad_W_i_, grad_U_i_, grad_b_i_);

  // Compute gradients for forget gate
  compute_gate_gradients(dL_dgate_f, W_f_, U_f_,
                         grad_input, grad_h_prev,
                         grad_W_f_, grad_U_f_, grad_b_f_);

  // Accumulate gradients from hidden state gradients
  // Not handling gradients from future time steps in this implementation

  // If handling sequences, you would add grad_h_prev to gradients for the previous time step

  return grad_input;
}

// Update weights and biases using gradients
void LSTM::updateWeights(double learningRate) {
  // Helper lambda to update weights and biases
  auto update_params = [&](std::vector<std::vector<double>>& W,
                           std::vector<std::vector<double>>& U,
                           std::vector<double>& b,
                           std::vector<std::vector<double>>& grad_W,
                           std::vector<std::vector<double>>& grad_U,
                           std::vector<double>& grad_b) {
      for(size_t i = 0; i < W.size(); ++i) {
        for(size_t j = 0; j < W[0].size(); ++j) {
          W[i][j] -= learningRate * grad_W[i][j];
          grad_W[i][j] = 0.0; // Reset gradient
        }
      }
      for(size_t i = 0; i < U.size(); ++i) {
        for(size_t j = 0; j < U[0].size(); ++j) {
          U[i][j] -= learningRate * grad_U[i][j];
          grad_U[i][j] = 0.0; // Reset gradient
        }
      }
      for(size_t i = 0; i < b.size(); ++i) {
        b[i] -= learningRate * grad_b[i];
        grad_b[i] = 0.0; // Reset gradient
      }
  };

  // Update input gate parameters
  update_params(W_i_, U_i_, b_i_, grad_W_i_, grad_U_i_, grad_b_i_);

  // Update forget gate parameters
  update_params(W_f_, U_f_, b_f_, grad_W_f_, grad_U_f_, grad_b_f_);

  // Update cell gate parameters
  update_params(W_c_, U_c_, b_c_, grad_W_c_, grad_U_c_, grad_b_c_);

  // Update output gate parameters
  update_params(W_o_, U_o_, b_o_, grad_W_o_, grad_U_o_, grad_b_o_);
}

// Save the layer's parameters to a binary file
void LSTM::save(std::ofstream& ofs) const {
  // Save layer type identifier
  std::string layer_type = "LSTM";
  size_t type_length = layer_type.size();
  ofs.write(reinterpret_cast<const char*>(&type_length), sizeof(size_t));
  ofs.write(layer_type.c_str(), type_length);

  // Save input_size and hidden_size
  ofs.write(reinterpret_cast<const char*>(&input_size_), sizeof(size_t));
  ofs.write(reinterpret_cast<const char*>(&hidden_size_), sizeof(size_t));

  // Save weight matrices and biases for each gate

  // Input Gate: W_i, U_i, b_i
  for(const auto& row : W_i_) {
    ofs.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));
  }
  for(const auto& row : U_i_) {
    ofs.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));
  }
  ofs.write(reinterpret_cast<const char*>(b_i_.data()), b_i_.size() * sizeof(double));

  // Forget Gate: W_f, U_f, b_f
  for(const auto& row : W_f_) {
    ofs.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));
  }
  for(const auto& row : U_f_) {
    ofs.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));
  }
  ofs.write(reinterpret_cast<const char*>(b_f_.data()), b_f_.size() * sizeof(double));

  // Cell Gate: W_c, U_c, b_c
  for(const auto& row : W_c_) {
    ofs.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));
  }
  for(const auto& row : U_c_) {
    ofs.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));
  }
  ofs.write(reinterpret_cast<const char*>(b_c_.data()), b_c_.size() * sizeof(double));

  // Output Gate: W_o, U_o, b_o
  for(const auto& row : W_o_) {
    ofs.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));
  }
  for(const auto& row : U_o_) {
    ofs.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));
  }
  ofs.write(reinterpret_cast<const char*>(b_o_.data()), b_o_.size() * sizeof(double));
}

// Load the layer's parameters from a binary file
std::unique_ptr<Layer> LSTM::load(std::ifstream& ifs) {
  size_t input_size, hidden_size;
  ifs.read(reinterpret_cast<char*>(&input_size), sizeof(size_t));
  ifs.read(reinterpret_cast<char*>(&hidden_size), sizeof(size_t));

  auto layer = std::make_unique<LSTM>(input_size, hidden_size);

  // Load Input Gate: W_i, U_i, b_i
  for(auto& row : layer->W_i_) {
    ifs.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
  }
  for(auto& row : layer->U_i_) {
    ifs.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
  }
  ifs.read(reinterpret_cast<char*>(layer->b_i_.data()), layer->b_i_.size() * sizeof(double));

  // Load Forget Gate: W_f, U_f, b_f
  for(auto& row : layer->W_f_) {
    ifs.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
  }
  for(auto& row : layer->U_f_) {
    ifs.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
  }
  ifs.read(reinterpret_cast<char*>(layer->b_f_.data()), layer->b_f_.size() * sizeof(double));

  // Load Cell Gate: W_c, U_c, b_c
  for(auto& row : layer->W_c_) {
    ifs.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
  }
  for(auto& row : layer->U_c_) {
    ifs.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
  }
  ifs.read(reinterpret_cast<char*>(layer->b_c_.data()), layer->b_c_.size() * sizeof(double));

  // Load Output Gate: W_o, U_o, b_o
  for(auto& row : layer->W_o_) {
    ifs.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
  }
  for(auto& row : layer->U_o_) {
    ifs.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
  }
  ifs.read(reinterpret_cast<char*>(layer->b_o_.data()), layer->b_o_.size() * sizeof(double));

  return layer;
}
