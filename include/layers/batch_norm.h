#ifndef QUANT_TRADING_DNN_BATCH_NORM_H
#define QUANT_TRADING_DNN_BATCH_NORM_H

#include "layer.h"
#include <vector>

class BatchNorm : public Layer {
public:
    explicit BatchNorm(size_t num_features, double momentum = 0.9, double epsilon = 1e-5);
    std::vector<double> forward(const std::vector<double>& input) override;
    std::vector<double> backward(const std::vector<double>& gradOutput) override;
    void updateWeights(double learningRate) override;
    void save(std::ofstream& ofs) const override;
    static std::unique_ptr<Layer> load(std::ifstream& ifs);

private:
    size_t num_features_;
    double momentum_;
    double epsilon_;

    std::vector<double> gamma_;
    std::vector<double> beta_;
    std::vector<double> running_mean_;
    std::vector<double> running_var_;

    // Cache variables for backward pass
    std::vector<double> x_hat_;
    std::vector<double> mean_;
    std::vector<double> var_;
    std::vector<double> std_inv_;
    std::vector<double> grad_gamma_;
    std::vector<double> grad_beta_;
};


#endif //QUANT_TRADING_DNN_BATCH_NORM_H
