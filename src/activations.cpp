#include "utils/activations.h"

namespace activations {
    Tensor relu(const Tensor &input) {
      Tensor result = input.clone();
      std::transform(result.data().begin(), result.data().end(), result.data().begin(),
                     [](double val) { return val > 0 ? val : 0; });
      return result;
    }

    Tensor relu_derivative(const Tensor &input) {
      Tensor result(input.shape());
      std::transform(input.data().begin(), input.data().end(), result.data().begin(),
                     [](double val) { return val > 0 ? 1.0 : 0.0; });
      return result;
    }

    Tensor leaky_relu(const Tensor &input, double alpha) {
      Tensor result = input.clone();
      std::transform(result.data().begin(), result.data().end(), result.data().begin(),
                     [alpha](double val) { return val > 0 ? val : alpha * val; });
      return result;
    }

    Tensor leaky_relu_derivative(const Tensor &input, double alpha) {
      Tensor result(input.shape());
      std::transform(input.data().begin(), input.data().end(), result.data().begin(),
                     [alpha](double val) { return val > 0 ? 1.0 : alpha; });
      return result;
    }

    Tensor parametric_relu(const Tensor &input, double alpha) {
      Tensor result = input.clone();
      std::transform(result.data().begin(), result.data().end(), result.data().begin(),
                     [alpha](double val) { return val > 0 ? val : alpha * val; });
      return result;
    }

    Tensor parametric_relu_derivative(const Tensor &input, double alpha) {
      Tensor result(input.shape());
      std::transform(input.data().begin(), input.data().end(), result.data().begin(),
                     [alpha](double val) { return val > 0 ? 1.0 : alpha; });
      return result;
    }

    Tensor elu(const Tensor &input, double alpha) {
      Tensor result = input.clone();
      std::transform(result.data().begin(), result.data().end(), result.data().begin(),
                     [alpha](double val) { return val > 0 ? val : alpha * (std::exp(val) - 1); });
      return result;
    }

    Tensor elu_derivative(const Tensor &input, double alpha) {
      Tensor result(input.shape());
      std::transform(input.data().begin(), input.data().end(), result.data().begin(),
                     [alpha](double val) { return val > 0 ? 1.0 : alpha * std::exp(val); });
      return result;
    }

    Tensor selu(const Tensor &input) {
      const double alpha = 1.6732632423543772848170429916717;
      const double scale = 1.0507009873554804934193349852946;
      Tensor result = input.clone();
      std::transform(result.data().begin(), result.data().end(), result.data().begin(),
                     [alpha, scale](double val) { return val > 0 ? scale * val : scale * (alpha * std::exp(val) - alpha); });
      return result;
    }

    Tensor selu_derivative(const Tensor &input) {
      const double alpha = 1.6732632423543772848170429916717;
      const double scale = 1.0507009873554804934193349852946;
      Tensor result(input.shape());
      std::transform(input.data().begin(), input.data().end(), result.data().begin(),
                     [alpha, scale](double val) { return val > 0 ? scale : scale * alpha * std::exp(val); });
      return result;
    }

    Tensor sigmoid(const Tensor &input) {
      Tensor result = input.clone();
      std::transform(result.data().begin(), result.data().end(), result.data().begin(),
                     [](double val) { return 1.0 / (1.0 + std::exp(-val)); });
      return result;
    }

    Tensor sigmoid_derivative(const Tensor &input) {
      Tensor result = sigmoid(input);
      result = result * (1 - result);
      return result;
    }

    Tensor tanh(const Tensor &input) {
      Tensor result = input.clone();
      std::transform(result.data().begin(), result.data().end(), result.data().begin(),
                     [](double val) { return std::tanh(val); });
      return result;
    }

    Tensor tanh_derivative(const Tensor &input) {
      Tensor result = tanh(input);
      result = 1 - result * result;
      return result;
    }

    Tensor hard_sigmoid(const Tensor &input) {
      Tensor result = input.clone();
      std::transform(result.data().begin(), result.data().end(), result.data().begin(),
                     [](double val) {
                       if (val <= -2.5) return 0.0;
                       if (val >= 2.5) return 1.0;
                       return 0.2 * val + 0.5;
                     });
      return result;
    }

    Tensor hard_sigmoid_derivative(const Tensor &input) {
      Tensor result(input.shape());
      std::transform(input.data().begin(), input.data().end(), result.data().begin(),
                     [](double val) {
                       if (val <= -2.5 || val >= 2.5) return 0.0;
                       return 0.2;
                     });
      return result;
    }

    Tensor hard_tanh(const Tensor &input) {
      Tensor result = input.clone();
      std::transform(result.data().begin(), result.data().end(), result.data().begin(),
                     [](double val) {
                       if (val <= -1.0) return -1.0;
                       if (val >= 1.0) return 1.0;
                       return val;
                     });
      return result;
    }

    Tensor hard_tanh_derivative(const Tensor &input) {
      Tensor result(input.shape());
      std::transform(input.data().begin(), input.data().end(), result.data().begin(),
                     [](double val) {
                       if (val <= -1.0 || val >= 1.0) return 0.0;
                       return 1.0;
                     });
      return result;
    }

    Tensor softmax(const Tensor &input, int axis = -1) {
      if (axis < 0) {
        axis += static_cast<int>(input.shape().size());
      }
      if (axis < 0 || axis >= static_cast<int>(input.shape().size())) {
        throw std::invalid_argument("Axis out of bounds for softmax.");
      }

      // Calculate the total number of elements in the reduced dimension
      size_t reduce_dim = input.shape()[axis];

      // Compute strides
      std::vector<size_t> strides(input.shape().size(), 1);
      for (int i = static_cast<int>(input.shape().size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * input.shape()[i + 1];
      }

      Tensor result(input.shape());
      size_t outer_dim = input.size() / (reduce_dim * strides[axis]);

      for (size_t outer = 0; outer < outer_dim; ++outer) {
        for (size_t inner = 0; inner < strides[axis]; ++inner) {
          // Find the max value for numerical stability
          double max_val = -std::numeric_limits<double>::infinity();
          for (size_t i = 0; i < reduce_dim; ++i) {
            size_t idx = outer * reduce_dim * strides[axis] + i * strides[axis] + inner;
            if (input.data()[idx] > max_val) {
              max_val = input.data()[idx];
            }
          }
          // Compute exponentials and sum
          double sum_exp = 0.0;
          for (size_t i = 0; i < reduce_dim; ++i) {
            size_t idx = outer * reduce_dim * strides[axis] + i * strides[axis] + inner;
            result.data()[idx] = std::exp(input.data()[idx] - max_val);
            sum_exp += result.data()[idx];
          }
          // Normalize
          for (size_t i = 0; i < reduce_dim; ++i) {
            size_t idx = outer * reduce_dim * strides[axis] + i * strides[axis] + inner;
            result.data()[idx] /= sum_exp;
          }
        }
      }
      return result;
    }

    Tensor softmax_derivative(const Tensor &input, int axis = -1) {
      Tensor result = softmax(input, axis);
      result = result * (1 - result);
      return result;
    }

    Tensor softplus(const Tensor &input) {
      Tensor result = input.clone();
      std::transform(result.data().begin(), result.data().end(), result.data().begin(),
                     [](double val) { return std::log1p(std::exp(val)); });
      return result;
    }

    Tensor softplus_derivative(const Tensor &input) {
      Tensor result = input.clone();
      std::transform(result.data().begin(), result.data().end(), result.data().begin(),
                     [](double val) { return 1.0 / (1.0 + std::exp(-val)); });
      return result;
    }

    Tensor softsign(const Tensor &input) {
      Tensor result = input.clone();
      std::transform(result.data().begin(), result.data().end(), result.data().begin(),
                     [](double val) { return val / (1.0 + std::abs(val)); });
      return result;
    }

    Tensor softsign_derivative(const Tensor &input) {
      Tensor result = input.clone();
      std::transform(result.data().begin(), result.data().end(), result.data().begin(),
                     [](double val) {
                       double abs_val = std::abs(val);
                       return 1.0 / ((1.0 + abs_val) * (1.0 + abs_val));
                     });
      return result;
    }

    Tensor swish(const Tensor &input) {
      Tensor result = input.clone();
      std::transform(result.data().begin(), result.data().end(), result.data().begin(),
                     [](double val) { return val / (1.0 + std::exp(-val)); });
      return result;
    }

    Tensor swish_derivative(const Tensor &input) {
      Tensor result = input.clone();
      std::transform(result.data().begin(), result.data().end(), result.data().begin(),
                     [](double val) {
                       double exp_val = std::exp(val);
                       double sigmoid_val = 1.0 / (1.0 + exp_val);
                       return sigmoid_val + val * exp_val * sigmoid_val * sigmoid_val;
                     });
      return result;
    }

    Tensor gelu(const Tensor &input) {
      Tensor result = input.clone();
      std::transform(result.data().begin(), result.data().end(), result.data().begin(),
                     [](double val) {
                       return 0.5 * val * (1.0 + std::tanh(std::sqrt(2.0 / M_PI) * (val + 0.044715 * val * val * val)));
                     });
      return result;
    }

    Tensor gelu_derivative(const Tensor &input) {
      Tensor result = input.clone();
      std::transform(result.data().begin(), result.data().end(), result.data().begin(),
                     [](double val) {
                       double cdf = 0.5 * (1.0 + std::tanh(std::sqrt(2.0 / M_PI) * (val + 0.044715 * val * val * val)));
                       double pdf = std::exp(-0.5 * val * val) / std::sqrt(2.0 * M_PI);
                       return 0.5 * (1.0 + cdf + val * pdf * (1.0 - cdf));
                     });
      return result;
    }

    Tensor maxout(const Tensor &input, int num_linear) {
      if (input.shape().back() % num_linear != 0) {
        throw std::invalid_argument("Number of linear pieces must divide the last dimension of the input.");
      }
      Tensor result(input.shape());
      size_t num_features = input.shape().back() / num_linear;
      for (size_t i = 0; i < input.size(); i += num_features) {
        for (size_t j = 0; j < num_features; ++j) {
          double max_val = input.data()[i + j];
          for (int k = 1; k < num_linear; ++k) {
            max_val = std::max(max_val, input.data()[i + k * num_features + j]);
          }
          for (int k = 0; k < num_linear; ++k) {
            result.data()[i + k * num_features + j] = max_val;
          }
        }
      }
      return result;
    }

    Tensor maxout_derivative(const Tensor &input, int num_linear) {
      if (input.shape().back() % num_linear != 0) {
        throw std::invalid_argument("Number of linear pieces must divide the last dimension of the input.");
      }
      Tensor result(input.shape());
      size_t num_features = input.shape().back() / num_linear;
      for (size_t i = 0; i < input.size(); i += num_features) {
        for (size_t j = 0; j < num_features; ++j) {
          double max_val = input.data()[i + j];
          int max_idx = 0;
          for (int k = 1; k < num_linear; ++k) {
            if (input.data()[i + k * num_features + j] > max_val) {
              max_val = input.data()[i + k * num_features + j];
              max_idx = k;
            }
          }
          for (int k = 0; k < num_linear; ++k) {
            result.data()[i + k * num_features + j] = k == max_idx ? 1.0 : 0.0;
          }
        }
      }
      return result;
    }

    Tensor mish(const Tensor &input) {
      Tensor result = input.clone();
      std::transform(result.data().begin(), result.data().end(), result.data().begin(),
                     [](double val) {
                       double exp_val = std::exp(val);
                       double exp_2val = std::exp(2.0 * val);
                       return val * std::tanh(std::log(1.0 + exp_val)) * (4.0 * exp_2val + 4.0 * exp_val + exp_val * val + val + 1.0) / ((exp_2val + 2.0 * exp_val + 1.0) * (exp_2val + 2.0 * exp_val + 1.0));
                     });
      return result;
    }

    Tensor mish_derivative(const Tensor &input) {
      // TODO: Impliment this
    }

    Tensor thresholded_relu(const Tensor &input, double theta) {
      Tensor result = input.clone();
      std::transform(result.data().begin(), result.data().end(), result.data().begin(),
                     [theta](double val) { return val > theta ? val : 0.0; });
      return result;
    }

    Tensor thresholded_relu_derivative(const Tensor &input, double theta) {
      Tensor result(input.shape());
      std::transform(input.data().begin(), input.data().end(), result.data().begin(),
                     [theta](double val) { return val > theta ? 1.0 : 0.0; });
      return result;
    }

    Tensor apl(const Tensor &input, const std::vector<double> &params) {
      if (params.size() != 3) {
        throw std::invalid_argument("APL activation requires 3 parameters.");
      }
      Tensor result = input.clone();
      std::transform(result.data().begin(), result.data().end(), result.data().begin(),
                     [params](double val) {
                       return params[0] * val + params[1] * std::tanh(params[2] * val);
                     });
      return result;
    }

    Tensor apl_derivative(const Tensor &input, const std::vector<double> &params) {
      if (params.size() != 3) {
        throw std::invalid_argument("APL activation requires 3 parameters.");
      }
      Tensor result(input.shape());
      std::transform(input.data().begin(), input.data().end(), result.data().begin(),
                     [params](double val) {
                       double tanh_val = std::tanh(params[2] * val);
                       return params[0] + params[1] * (1.0 - tanh_val * tanh_val) * params[2];
                     });
      return result;
    }

    Tensor bent_identity(const Tensor &input) {
      Tensor result = input.clone();
      std::transform(result.data().begin(), result.data().end(), result.data().begin(),
                     [](double val) {
                       return (std::sqrt(val * val + 1.0) - 1.0) / 2.0 + val;
                     });
      return result;
    }

    Tensor bent_identity_derivative(const Tensor &input) {
      Tensor result = input.clone();
      std::transform(result.data().begin(), result.data().end(), result.data().begin(),
                     [](double val) {
                       return val / (2.0 * std::sqrt(val * val + 1.0)) + 1.0;
                     });
      return result;
    }

    Tensor eswish(const Tensor &input, double beta) {
      Tensor result = input.clone();
      std::transform(result.data().begin(), result.data().end(), result.data().begin(),
                     [beta](double val) {
                       return val * std::tanh(beta * val);
                     });
      return result;
    }

    Tensor eswish_derivative(const Tensor &input, double beta) {
      Tensor result = input.clone();
      std::transform(result.data().begin(), result.data().end(), result.data().begin(),
                     [beta](double val) {
                       double tanh_val = std::tanh(beta * val);
                       return tanh_val + beta * val * (1.0 - tanh_val * tanh_val);
                     });
      return result;
    }

    Tensor log_sigmoid(const Tensor &input) {
      Tensor result = input.clone();
      std::transform(result.data().begin(), result.data().end(), result.data().begin(),
                     [](double val) {
                       return -std::log1p(std::exp(-val));
                     });
      return result;
    }

    Tensor log_sigmoid_derivative(const Tensor &input) {
      Tensor result = input.clone();
      std::transform(result.data().begin(), result.data().end(), result.data().begin(),
                     [](double val) {
                       return -1.0 / (1.0 + std::exp(val));
                     });
      return result;
    }

    Tensor sinc(const Tensor &input) {
      Tensor result = input.clone();
      std::transform(result.data().begin(), result.data().end(), result.data().begin(),
                     [](double val) {
                       if (val == 0.0) {
                         return 1.0;
                       }
                       return std::sin(val) / val;
                     });
      return result;
    }

    Tensor sinc_derivative(const Tensor &input) {
      Tensor result = input.clone();
      std::transform(result.data().begin(), result.data().end(), result.data().begin(),
                     [](double val) {
                       if (val == 0.0) {
                         return 0.0;
                       }
                       return (val * std::cos(val) - std::sin(val)) / (val * val);
                     });
      return result;
    }

    Tensor tanh_exp(const Tensor &input) {
      Tensor result = input.clone();
      std::transform(result.data().begin(), result.data().end(), result.data().begin(),
                     [](double val) {
                       return val * std::exp(val);
                     });
      return result;
    }

    Tensor tanh_exp_derivative(const Tensor &input) {
      Tensor result = input.clone();
      std::transform(result.data().begin(), result.data().end(), result.data().begin(),
                     [](double val) {
                       return std::exp(val) + val * std::exp(val);
                     });
      return result;
    }



}