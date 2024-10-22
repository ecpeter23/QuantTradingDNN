#ifndef QUANT_TRADING_DNN_CONV1D_H
#define QUANT_TRADING_DNN_CONV1D_H

#include "layer.h"
#include <vector>

class Conv1D : public Layer {
public:
    Conv1D(size_t input_channels, size_t output_channels, size_t kernel_size, size_t stride=1, size_t padding=0);
    std::vector<double> forward(const std::vector<double>& input) override;
    std::vector<double> backward(const std::vector<double>& gradOutput) override;
    void updateWeights(double learningRate) override;
    void save(std::ofstream& ofs) const override;
    static std::unique_ptr<Layer> load(std::ifstream& ifs);

private:
    size_t input_channels_;
    size_t output_channels_;
    size_t kernel_size_;
    size_t stride_;
    size_t padding_;

    // Weight tensors: [output_channels][input_channels][kernel_size]
    std::vector<std::vector<std::vector<double>>> weights_;
    std::vector<double> biases_;

    // Gradients
    std::vector<std::vector<std::vector<double>>> grad_weights_;
    std::vector<double> grad_biases_;

    // Caching for backward pass
    std::vector<std::vector<double>> input_channels_padded_;
    size_t input_length_; // Length of the input sequence after padding

    // Utility functions
    void padInput(const std::vector<double>& input);
};

#endif //QUANT_TRADING_DNN_CONV1D_H
