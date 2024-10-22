#ifndef QUANT_TRADING_DNN_LSTM_H
#define QUANT_TRADING_DNN_LSTM_H

#include "layer.h"
#include <vector>
#include <memory>

class LSTM : public Layer {
public:
    LSTM(size_t input_size, size_t hidden_size);
    std::vector<double> forward(const std::vector<double>& input) override;
    std::vector<double> backward(const std::vector<double>& gradOutput) override;
    void updateWeights(double learningRate) override;
    void save(std::ofstream& ofs) const override;
    static std::unique_ptr<Layer> load(std::ifstream& ifs);

private:
    size_t input_size_;
    size_t hidden_size_;

    // Weight matrices for input gates
    std::vector<std::vector<double>> W_i_; // [hidden_size_][input_size_]
    std::vector<std::vector<double>> U_i_; // [hidden_size_][hidden_size_]
    std::vector<double> b_i_;              // [hidden_size_]

    // Weight matrices for forget gates
    std::vector<std::vector<double>> W_f_; // [hidden_size_][input_size_]
    std::vector<std::vector<double>> U_f_; // [hidden_size_][hidden_size_]
    std::vector<double> b_f_;              // [hidden_size_]

    // Weight matrices for cell gates
    std::vector<std::vector<double>> W_c_; // [hidden_size_][input_size_]
    std::vector<std::vector<double>> U_c_; // [hidden_size_][hidden_size_]
    std::vector<double> b_c_;              // [hidden_size_]

    // Weight matrices for output gates
    std::vector<std::vector<double>> W_o_; // [hidden_size_][input_size_]
    std::vector<std::vector<double>> U_o_; // [hidden_size_][hidden_size_]
    std::vector<double> b_o_;              // [hidden_size_]

    // Gradients for weights and biases
    std::vector<std::vector<double>> grad_W_i_;
    std::vector<std::vector<double>> grad_U_i_;
    std::vector<double> grad_b_i_;

    std::vector<std::vector<double>> grad_W_f_;
    std::vector<std::vector<double>> grad_U_f_;
    std::vector<double> grad_b_f_;

    std::vector<std::vector<double>> grad_W_c_;
    std::vector<std::vector<double>> grad_U_c_;
    std::vector<double> grad_b_c_;

    std::vector<std::vector<double>> grad_W_o_;
    std::vector<std::vector<double>> grad_U_o_;
    std::vector<double> grad_b_o_;

    // Cell state and hidden state
    std::vector<double> c_; // [hidden_size_]
    std::vector<double> h_; // [hidden_size_]

    // Previous cell state and hidden state for backpropagation
    std::vector<double> c_prev_; // [hidden_size_]
    std::vector<double> h_prev_; // [hidden_size_]

    // Gate activations and intermediates for backpropagation
    std::vector<double> i_; // Input gate
    std::vector<double> f_; // Forget gate
    std::vector<double> g_; // Cell gate
    std::vector<double> o_; // Output gate

    // Cached input for backward pass
    std::vector<double> input_; // [input_size_]

    // Activation functions
    double sigmoid(double x) const;
    double tanh_func(double x) const;
};

#endif //QUANT_TRADING_DNN_LSTM_H
