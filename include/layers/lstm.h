#ifndef QUANT_TRADING_DNN_LSTM_H
#define QUANT_TRADING_DNN_LSTM_H

#include "layer.h"
#include <vector>

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

    // Weight matrices and biases for input, forget, cell, and output gates
    std::vector<std::vector<double>> W_i_;
    std::vector<double> b_i_;
    std::vector<std::vector<double>> W_f_;
    std::vector<double> b_f_;
    std::vector<std::vector<double>> W_c_;
    std::vector<double> b_c_;
    std::vector<std::vector<double>> W_o_;
    std::vector<double> b_o_;

    // Cell state and hidden state
    std::vector<double> c_;
    std::vector<double> h_;

    // Gradients
    // ... (gradients for weights and biases)

    // Activation functions
    double sigmoid(double x);
    double tanh_func(double x);
};



#endif //QUANT_TRADING_DNN_LSTM_H
