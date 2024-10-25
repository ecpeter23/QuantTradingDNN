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
      Tensor sigmoid_result = sigmoid(input);
      Tensor result = sigmoid_result * (1.0 - sigmoid_result);
      return result;
    }


    Tensor tanh(const Tensor &input) {
      Tensor result = input.clone();
      std::transform(result.data().begin(), result.data().end(), result.data().begin(),
                     [](double val) { return std::tanh(val); });
      return result;
    }

    Tensor tanh_derivative(const Tensor &input) {
      Tensor tanh_result = tanh(input);
      Tensor result = 1.0 - (tanh_result * tanh_result);
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

      // Move the axis to the last dimension
      std::vector<size_t> permuted_axes;
      for (size_t i = 0; i < input.shape().size(); ++i) {
        if (i != static_cast<size_t>(axis)) {
          permuted_axes.push_back(i);
        }
      }
      permuted_axes.push_back(static_cast<size_t>(axis));

      Tensor permuted_input = input.permute(permuted_axes);
      size_t reduce_dim = input.shape()[axis];
      size_t outer_dim = input.size() / reduce_dim;

      Tensor result = permuted_input.clone();

      for (size_t i = 0; i < outer_dim; ++i) {
        // Get the start of the current slice
        double* start = &result.data()[i * reduce_dim];

        // Find max for numerical stability
        double max_val = *std::max_element(start, start + reduce_dim);

        // Compute exponentials and sum
        double sum_exp = 0.0;
        for (size_t j = 0; j < reduce_dim; ++j) {
          start[j] = std::exp(start[j] - max_val);
          sum_exp += start[j];
        }

        // Normalize
        for (size_t j = 0; j < reduce_dim; ++j) {
          start[j] /= sum_exp;
        }
      }

      // Permute back to original axes
      // Generate inverse permutation
      std::vector<size_t> inverse_permuted_axes(input.shape().size());
      for (size_t i = 0; i < permuted_axes.size(); ++i) {
        inverse_permuted_axes[permuted_axes[i]] = i;
      }
      return result.permute(inverse_permuted_axes);
    }

    Tensor softmax_derivative(const Tensor &input, int axis = -1) {
      Tensor s = softmax(input, axis);

      // Assuming input is a 1D tensor for simplicity
      if (input.shape().size() != 1) {
        throw std::invalid_argument("Softmax derivative implementation assumes input is a 1D tensor.");
      }
      size_t n = input.size();
      Tensor jacobian({n, n});

      for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
          if (i == j) {
            jacobian({i, j}) = s.data()[i] * (1 - s.data()[i]);
          } else {
            jacobian({i, j}) = -s.data()[i] * s.data()[j];
          }
        }
      }
      return jacobian;
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
      Tensor sigmoid_result = sigmoid(input);
      Tensor term1 = sigmoid_result;
      Tensor term2 = input * sigmoid_result * (1.0 - sigmoid_result);
      Tensor result = term1 + term2;
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
      Tensor result(input.shape());
      const double sqrt_2_over_pi = std::sqrt(2.0 / M_PI);
      std::transform(input.data().begin(), input.data().end(), result.data().begin(),
                     [sqrt_2_over_pi](double x) {
                         double tanh_arg = sqrt_2_over_pi * (x + 0.044715 * std::pow(x, 3));
                         double tanh_val = std::tanh(tanh_arg);
                         double left = 0.5 * tanh_val + 0.5;
                         double sech2 = 1 - tanh_val * tanh_val;
                         double right = 0.5 * sqrt_2_over_pi * (1 + 3 * 0.044715 * x * x) * sech2;
                         return left + x * right;
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
      Tensor result(input.shape());
      std::transform(input.data().begin(), input.data().end(), result.data().begin(),
                     [](double x) {
                         double sp = std::log1p(std::exp(x)); // Softplus(x)
                         return x * std::tanh(sp);
                     });
      return result;
    }

    Tensor mish_derivative(const Tensor &input) {
      Tensor result(input.shape());
      std::transform(input.data().begin(), input.data().end(), result.data().begin(),
                     [](double x) {
                         double sp = std::log1p(std::exp(x)); // Softplus(x)
                         double tanh_sp = std::tanh(sp);
                         double sigmoid_x = 1.0 / (1.0 + std::exp(-x)); // Sigmoid(x)
                         double derivative = tanh_sp + x * sigmoid_x * (1.0 - tanh_sp * tanh_sp);
                         return derivative;
                     });
      return result;
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