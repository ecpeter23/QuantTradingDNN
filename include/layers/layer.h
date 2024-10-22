#ifndef QUANT_TRADING_DNN_LAYER_H
#define QUANT_TRADING_DNN_LAYER_H

#include <vector>
#include <memory>
#include <fstream>
#include <string>
#include <variant>

/**
 * Base class for all neural network layers.
 */
class Layer {
protected:
    bool is_training_ = false;  // Training mode flag

public:
    virtual ~Layer() = default;

    // Set whether the layer is in training mode
    void setTrainingMode(bool is_training) {
      is_training_ = is_training;
    }

    /**
     * Generalized forward pass using variant to support 1D, 2D, 3D, or 4D tensors.
     */
    using Tensor = std::variant<std::vector<double>, std::vector<std::vector<double>>,
            std::vector<std::vector<std::vector<double>>>,
            std::vector<std::vector<std::vector<std::vector<double>>>>>;

    virtual Tensor forward(const Tensor& input) = 0;

    virtual Tensor backward(const Tensor& gradOutput) = 0;

    // Update weights (if the layer has any)
    virtual void updateWeights(double learningRate) = 0;

    // Save the layer’s parameters (for model serialization)
    virtual void save(std::ofstream& ofs) const = 0;

    // Load layer parameters from file
    static std::unique_ptr<Layer> load(std::ifstream& ifs);

    // Helper to identify layer type (for debugging/logging)
    [[nodiscard]] virtual std::string layerType() const = 0;
};

#endif // QUANT_TRADING_DNN_LAYER_H
