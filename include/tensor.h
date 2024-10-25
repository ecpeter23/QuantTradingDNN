#ifndef QUANT_TRADING_DNN_TENSOR_H
#define QUANT_TRADING_DNN_TENSOR_H

#include <vector>
#include <cstddef>
#include <stdexcept>
#include <numeric>
#include <functional>
#include <initializer_list>
#include <algorithm>
#include <Accelerate/Accelerate.h>
#include <cmath>
#include <limits>
#include <random>

class Tensor {
public:
    // Default constructor
    Tensor() = default;

    // Constructor with shape, initializes data to zeros
    explicit Tensor(const std::vector<size_t>& shape)
            : shape_(shape), size_(calculateSize(shape)), data_(size_, 0.0) {}

    // Constructor with shape and initial value
    Tensor(const std::vector<size_t>& shape, double init_value)
            : shape_(shape), size_(calculateSize(shape)), data_(size_, init_value) {}

    // Constructor with shape and data
    Tensor(const std::vector<size_t>& shape, const std::vector<double>& data)
            : shape_(shape), size_(calculateSize(shape)), data_(data) {
      if (data_.size() != size_) {
        throw std::invalid_argument("Data size does not match the shape size.");
      }
    }

    // Accessors for const and non-const contexts
    double& operator()(std::initializer_list<size_t> indices) {
      size_t flat_index = computeFlatIndex(indices);
      return data_[flat_index];
    }

    const double& operator()(std::initializer_list<size_t> indices) const {
      size_t flat_index = computeFlatIndex(indices);
      return data_[flat_index];
    }

    // Getters for shape and size
    [[nodiscard]] const std::vector<size_t>& shape() const {
      return shape_;
    }

    [[nodiscard]] size_t size() const {
      return size_;
    }

    // Data access
    [[nodiscard]] const std::vector<double>& data() const {
      return data_;
    }

    std::vector<double>& data() {
      return data_;
    }

    // Reshape tensor (must have the same total size)
    void reshape(const std::vector<size_t>& new_shape) {
      size_t new_size = calculateSize(new_shape);
      if (new_size != size_) {
        throw std::invalid_argument("New shape must have the same total size.");
      }
      shape_ = new_shape;
    }

    // Fill tensor with a value
    void fill(double value) {
      std::fill(data_.begin(), data_.end(), value);
    }

    // Element-wise addition
    Tensor operator+(const Tensor& other) const {
      checkShapeCompatibility(other);
      Tensor result(shape_);
      std::transform(data_.begin(), data_.end(), other.data_.begin(),
                     result.data_.begin(), std::plus<>());
      return result;
    }

    Tensor& operator+=(const Tensor& other) {
      checkShapeCompatibility(other);
      std::transform(data_.begin(), data_.end(), other.data_.begin(),
                     data_.begin(), std::plus<>());
      return *this;
    }

    // Element-wise subtraction
    Tensor operator-(const Tensor& other) const {
      checkShapeCompatibility(other);
      Tensor result(shape_);
      std::transform(data_.begin(), data_.end(), other.data_.begin(),
                     result.data_.begin(), std::minus<>());
      return result;
    }

    Tensor& operator-=(const Tensor& other) {
      checkShapeCompatibility(other);
      std::transform(data_.begin(), data_.end(), other.data_.begin(),
                     data_.begin(), std::minus<>());
      return *this;
    }

    // Element-wise multiplication
    Tensor operator*(const Tensor& other) const {
      checkShapeCompatibility(other);
      Tensor result(shape_);
      std::transform(data_.begin(), data_.end(), other.data_.begin(),
                     result.data_.begin(), std::multiplies<>());
      return result;
    }

    Tensor& operator*=(const Tensor& other) {
      checkShapeCompatibility(other);
      std::transform(data_.begin(), data_.end(), other.data_.begin(),
                     data_.begin(), std::multiplies<>());
      return *this;
    }

    // Element-wise division
    Tensor operator/(const Tensor& other) const {
      checkShapeCompatibility(other);
      Tensor result(shape_);
      std::transform(data_.begin(), data_.end(), other.data_.begin(),
                     result.data_.begin(), std::divides<>());
      return result;
    }

    Tensor& operator/=(const Tensor& other) {
      checkShapeCompatibility(other);
      std::transform(data_.begin(), data_.end(), other.data_.begin(),
                     data_.begin(), std::divides<>());
      return *this;
    }

    // Scalar operations
    Tensor operator+(double scalar) const {
      Tensor result(shape_);
      std::transform(data_.begin(), data_.end(), result.data_.begin(),
                     [scalar](double val) { return val + scalar; });
      return result;
    }

    Tensor& operator+=(double scalar) {
      std::transform(data_.begin(), data_.end(), data_.begin(),
                     [scalar](double val) { return val + scalar; });
      return *this;
    }

    Tensor operator-(double scalar) const {
      Tensor result(shape_);
      std::transform(data_.begin(), data_.end(), result.data_.begin(),
                     [scalar](double val) { return val - scalar; });
      return result;
    }

    Tensor& operator-=(double scalar) {
      std::transform(data_.begin(), data_.end(), data_.begin(),
                     [scalar](double val) { return val - scalar; });
      return *this;
    }

    Tensor operator*(double scalar) const {
      Tensor result(shape_);
      std::transform(data_.begin(), data_.end(), result.data_.begin(),
                     [scalar](double val) { return val * scalar; });
      return result;
    }

    Tensor& operator*=(double scalar) {
      std::transform(data_.begin(), data_.end(), data_.begin(),
                     [scalar](double val) { return val * scalar; });
      return *this;
    }

    Tensor operator/(double scalar) const {
      Tensor result(shape_);
      std::transform(data_.begin(), data_.end(), result.data_.begin(),
                     [scalar](double val) { return val / scalar; });
      return result;
    }

    Tensor& operator/=(double scalar) {
      std::transform(data_.begin(), data_.end(), data_.begin(),
                     [scalar](double val) { return val / scalar; });
      return *this;
    }

    // Activation functions
    [[nodiscard]] Tensor relu() const {
      Tensor result(shape_);
      std::transform(data_.begin(), data_.end(), result.data_.begin(),
                     [](double val) { return val > 0 ? val : 0; });
      return result;
    }

    [[nodiscard]] Tensor sigmoid() const {
      Tensor result(shape_);
      std::transform(data_.begin(), data_.end(), result.data_.begin(),
                     [](double val) { return 1.0 / (1.0 + std::exp(-val)); });
      return result;
    }

    [[nodiscard]] Tensor tanh() const {
      Tensor result(shape_);
      std::transform(data_.begin(), data_.end(), result.data_.begin(),
                     [](double val) { return std::tanh(val); });
      return result;
    }

    [[nodiscard]] Tensor leaky_relu(double alpha = 0.01) const {
      Tensor result(shape_);
      std::transform(data_.begin(), data_.end(), result.data_.begin(),
                     [alpha](double val) { return val > 0 ? val : alpha * val; });
      return result;
    }

    // Parametric ReLU (PReLU)
    [[nodiscard]] Tensor prelu(double alpha) const {
      Tensor result(shape_);
      std::transform(data_.begin(), data_.end(), result.data_.begin(),
                     [alpha](double val) { return val > 0 ? val : alpha * val; });
      return result;
    }

    // Exponential Linear Unit (ELU)
    [[nodiscard]] Tensor elu(double alpha = 1.0) const {
      Tensor result(shape_);
      std::transform(data_.begin(), data_.end(), result.data_.begin(),
                     [alpha](double val) { return val >= 0 ? val : alpha * (std::exp(val) - 1); });
      return result;
    }

    // Scaled Exponential Linear Unit (SELU)
    [[nodiscard]] Tensor selu() const {
      const double scale = 1.0507;
      const double alpha = 1.67326;
      Tensor result(shape_);
      std::transform(data_.begin(), data_.end(), result.data_.begin(),
                     [scale, alpha](double val) {
                         return val >= 0 ? scale * val : scale * alpha * (std::exp(val) - 1);
                     });
      return result;
    }

    // Hard Sigmoid
    [[nodiscard]] Tensor hard_sigmoid() const {
      Tensor result(shape_);
      std::transform(data_.begin(), data_.end(), result.data_.begin(),
                     [](double val) {
                         double res = 0.2 * val + 0.5;
                         res = std::min(std::max(res, 0.0), 1.0);
                         return res;
                     });
      return result;
    }

    // Hard Tanh
    [[nodiscard]] Tensor hard_tanh() const {
      Tensor result(shape_);
      std::transform(data_.begin(), data_.end(), result.data_.begin(),
                     [](double val) {
                         return std::min(std::max(val, -1.0), 1.0);
                     });
      return result;
    }

    // Softplus
    [[nodiscard]] Tensor softplus() const {
      Tensor result(shape_);
      std::transform(data_.begin(), data_.end(), result.data_.begin(),
                     [](double val) {
                         return std::log1p(std::exp(val));
                     });
      return result;
    }

    // Softsign
    [[nodiscard]] Tensor softsign() const {
      Tensor result(shape_);
      std::transform(data_.begin(), data_.end(), result.data_.begin(),
                     [](double val) {
                         return val / (1.0 + std::abs(val));
                     });
      return result;
    }

    // Swish
    [[nodiscard]] Tensor swish() const {
      Tensor result(shape_);
      std::transform(data_.begin(), data_.end(), result.data_.begin(),
                     [](double val) {
                         double sig = 1.0 / (1.0 + std::exp(-val));
                         return val * sig;
                     });
      return result;
    }

    // Gaussian Error Linear Unit (GELU)
    [[nodiscard]] Tensor gelu() const {
      Tensor result(shape_);
      const double sqrt_2_over_pi = std::sqrt(2.0 / M_PI);
      std::transform(data_.begin(), data_.end(), result.data_.begin(),
                     [sqrt_2_over_pi](double val) {
                         double cdf = 0.5 * (1.0 + std::tanh(sqrt_2_over_pi * (val + 0.044715 * std::pow(val, 3))));
                         return val * cdf;
                     });
      return result;
    }

    // Maxout
    [[nodiscard]] Tensor maxout(size_t num_units) const {
      if (shape_.empty()) {
        throw std::logic_error("Maxout requires at least one dimension.");
      }
      size_t last_dim = shape_.back();
      if (last_dim % num_units != 0) {
        throw std::invalid_argument("Last dimension size must be divisible by num_units.");
      }
      size_t new_last_dim = last_dim / num_units;
      std::vector<size_t> new_shape = shape_;
      new_shape.back() = new_last_dim;
      Tensor result(new_shape);

      size_t outer_size = size_ / last_dim;
      for (size_t i = 0; i < outer_size; ++i) {
        for (size_t j = 0; j < new_last_dim; ++j) {
          double max_val = data_[i * last_dim + j];
          for (size_t k = 1; k < num_units; ++k) {
            size_t idx = i * last_dim + j + k * new_last_dim;
            if (data_[idx] > max_val) {
              max_val = data_[idx];
            }
          }
          result.data_[i * new_last_dim + j] = max_val;
        }
      }
      return result;
    }

    // Mish
    [[nodiscard]] Tensor mish() const {
      Tensor result(shape_);
      std::transform(data_.begin(), data_.end(), result.data_.begin(),
                     [](double val) {
                         double sp = std::log1p(std::exp(val)); // Softplus
                         return val * std::tanh(sp);
                     });
      return result;
    }

    // Thresholded ReLU
    [[nodiscard]] Tensor thresholded_relu(double theta = 1.0) const {
      Tensor result(shape_);
      std::transform(data_.begin(), data_.end(), result.data_.begin(),
                     [theta](double val) { return val > theta ? val : 0.0; });
      return result;
    }

    // Adaptive Piecewise Linear (APL)
    [[nodiscard]] Tensor apl(const std::vector<double>& a, const std::vector<double>& b) const {
      if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors 'a' and 'b' must have the same size.");
      }
      Tensor result(shape_);
      size_t num_pieces = a.size();
      for (size_t idx = 0; idx < size_; ++idx) {
        double x = data_[idx];
        double sum = std::max(0.0, x);
        for (size_t i = 0; i < num_pieces; ++i) {
          sum += a[i] * std::max(0.0, -x + b[i]);
        }
        result.data_[idx] = sum;
      }
      return result;
    }

    // Bent Identity
    [[nodiscard]] Tensor bent_identity() const {
      Tensor result(shape_);
      std::transform(data_.begin(), data_.end(), result.data_.begin(),
                     [](double val) {
                         return ((std::sqrt(val * val + 1) - 1) / 2.0) + val;
                     });
      return result;
    }

    // ESwish
    [[nodiscard]] Tensor eswish(double beta) const {
      Tensor result(shape_);
      std::transform(data_.begin(), data_.end(), result.data_.begin(),
                     [beta](double val) {
                         double sig = 1.0 / (1.0 + std::exp(-val));
                         return beta * val * sig;
                     });
      return result;
    }

    // Log Sigmoid
    [[nodiscard]] Tensor log_sigmoid() const {
      Tensor result(shape_);
      std::transform(data_.begin(), data_.end(), result.data_.begin(),
                     [](double val) {
                         return -std::log1p(std::exp(-val));
                     });
      return result;
    }

    // Sinc
    [[nodiscard]] Tensor sinc() const {
      Tensor result(shape_);
      std::transform(data_.begin(), data_.end(), result.data_.begin(),
                     [](double val) {
                         return val == 0.0 ? 1.0 : std::sin(val) / val;
                     });
      return result;
    }

    // TanhExp
    [[nodiscard]] Tensor tanh_exp() const {
      Tensor result(shape_);
      std::transform(data_.begin(), data_.end(), result.data_.begin(),
                     [](double val) {
                         return val * std::tanh(std::exp(val));
                     });
      return result;
    }

    // Softmax along a specified axis (default is the last axis)
    [[nodiscard]] Tensor softmax(int axis = -1) const {
      if (axis < 0) {
        axis += static_cast<int>(shape_.size());
      }
      if (axis < 0 || axis >= static_cast<int>(shape_.size())) {
        throw std::invalid_argument("Axis out of bounds for softmax.");
      }

      // Calculate the total number of elements in the reduced dimension
      size_t reduce_dim = shape_[axis];

      // Compute strides
      std::vector<size_t> strides(shape_.size(), 1);
      for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape_[i + 1];
      }

      Tensor result(shape_);
      size_t outer_dim = size_ / (reduce_dim * strides[axis]);

      for (size_t outer = 0; outer < outer_dim; ++outer) {
        for (size_t inner = 0; inner < strides[axis]; ++inner) {
          // Find the max value for numerical stability
          double max_val = -std::numeric_limits<double>::infinity();
          for (size_t i = 0; i < reduce_dim; ++i) {
            size_t idx = outer * reduce_dim * strides[axis] + i * strides[axis] + inner;
            if (data_[idx] > max_val) {
              max_val = data_[idx];
            }
          }
          // Compute exponentials and sum
          double sum_exp = 0.0;
          for (size_t i = 0; i < reduce_dim; ++i) {
            size_t idx = outer * reduce_dim * strides[axis] + i * strides[axis] + inner;
            result.data_[idx] = std::exp(data_[idx] - max_val);
            sum_exp += result.data_[idx];
          }
          // Normalize
          for (size_t i = 0; i < reduce_dim; ++i) {
            size_t idx = outer * reduce_dim * strides[axis] + i * strides[axis] + inner;
            result.data_[idx] /= sum_exp;
          }
        }
      }
      return result;
    }

    // Reduction operations
    // Sum over all elements
    [[nodiscard]] double sum() const {
      return std::accumulate(data_.begin(), data_.end(), 0.0);
    }

    // Mean over all elements
    [[nodiscard]] double mean() const {
      return sum() / static_cast<double>(size_);
    }

    // Sum along a specific axis
    [[nodiscard]] Tensor sum(int axis) const {
      if (axis < 0) {
        axis += static_cast<int>(shape_.size());
      }
      if (axis < 0 || axis >= static_cast<int>(shape_.size())) {
        throw std::invalid_argument("Axis out of bounds for sum.");
      }

      std::vector<size_t> new_shape = shape_;
      new_shape.erase(new_shape.begin() + axis);

      Tensor result(new_shape, 0.0);

      // Compute strides
      std::vector<size_t> strides(shape_.size(), 1);
      for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape_[i + 1];
      }

      size_t reduce_dim = shape_[axis];
      size_t outer_dim = size_ / (reduce_dim * strides[axis]);

      for (size_t outer = 0; outer < outer_dim; ++outer) {
        for (size_t inner = 0; inner < strides[axis]; ++inner) {
          double sum_val = 0.0;
          for (size_t i = 0; i < reduce_dim; ++i) {
            size_t idx = outer * reduce_dim * strides[axis] + i * strides[axis] + inner;
            sum_val += data_[idx];
          }
          size_t out_idx = outer * strides[axis] + inner;
          result.data_[out_idx] = sum_val;
        }
      }
      return result;
    }

    // Mean along a specific axis
    [[nodiscard]] Tensor mean(int axis) const {
      Tensor sum_result = sum(axis);
      double divisor = static_cast<double>(shape_[axis]);
      std::transform(sum_result.data_.begin(), sum_result.data_.end(), sum_result.data_.begin(),
                     [divisor](double val) { return val / divisor; });
      return sum_result;
    }

    // Flatten the tensor
    [[nodiscard]] Tensor flatten() const {
      return Tensor({size_}, data_);
    }

    // Clone the tensor
    [[nodiscard]] Tensor clone() const {
      return Tensor(shape_, data_);
    }

    // Permute dimensions
    [[nodiscard]] Tensor permute(const std::vector<size_t>& dims) const {
      if (dims.size() != shape_.size()) {
        throw std::invalid_argument("Number of dimensions in permutation must match tensor dimensions.");
      }
      std::vector<size_t> new_shape(shape_.size());
      std::vector<size_t> new_strides(shape_.size());
      std::vector<size_t> old_strides(shape_.size());

      // Compute old strides
      old_strides[shape_.size() - 1] = 1;
      for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; --i) {
        old_strides[i] = old_strides[i + 1] * shape_[i + 1];
      }

      // Compute new shape and strides
      for (size_t i = 0; i < dims.size(); ++i) {
        if (dims[i] >= shape_.size()) {
          throw std::invalid_argument("Permutation indices are out of bounds.");
        }
        new_shape[i] = shape_[dims[i]];
      }
      // Compute new strides
      new_strides[new_shape.size() - 1] = 1;
      for (int i = static_cast<int>(new_shape.size()) - 2; i >= 0; --i) {
        new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
      }

      Tensor result(new_shape);

      // Reorder data
      for (size_t idx = 0; idx < size_; ++idx) {
        // Compute multi-dimensional index in old tensor
        size_t tmp = idx;
        std::vector<size_t> old_indices(shape_.size());
        for (size_t i = 0; i < shape_.size(); ++i) {
          old_indices[i] = tmp / old_strides[i];
          tmp %= old_strides[i];
        }
        // Compute new indices
        std::vector<size_t> new_indices(shape_.size());
        for (size_t i = 0; i < shape_.size(); ++i) {
          new_indices[i] = old_indices[dims[i]];
        }
        // Compute flat index in new tensor
        size_t new_idx = 0;
        for (size_t i = 0; i < new_shape.size(); ++i) {
          new_idx += new_indices[i] * new_strides[i];
        }
        result.data_[new_idx] = data_[idx];
      }
      return result;
    }

    // Random initialization (uniform distribution)
    void randu(double lower = 0.0, double upper = 1.0) {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<> dis(lower, upper);
      std::generate(data_.begin(), data_.end(), [&]() { return dis(gen); });
    }

    // Random initialization (normal distribution)
    void randn(double mean = 0.0, double stddev = 1.0) {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::normal_distribution<> dis(mean, stddev);
      std::generate(data_.begin(), data_.end(), [&]() { return dis(gen); });
    }

    // Xavier initialization (Glorot uniform)
    void xavier_uniform() {
      if (shape_.size() < 2) {
        throw std::logic_error("Xavier initialization requires at least 2D tensor.");
      }
      double limit = std::sqrt(6.0 / (shape_[0] + shape_[1]));
      randu(-limit, limit);
    }

    // Transpose (for 2D tensors)
    [[nodiscard]] Tensor transpose() const {
      if (shape_.size() != 2) {
        throw std::logic_error("Transpose is only defined for 2D tensors.");
      }
      Tensor result({shape_[1], shape_[0]});
      for (size_t i = 0; i < shape_[0]; ++i) {
        for (size_t j = 0; j < shape_[1]; ++j) {
          result({j, i}) = (*this)({i, j});
        }
      }
      return result;
    }

    // Matrix multiplication (supports batched matmul for N-dimensional tensors)
    [[nodiscard]] Tensor matmul(const Tensor& other) const {
      if (shape_.size() < 2 || other.shape_.size() < 2) {
        throw std::logic_error("Matmul requires tensors with at least 2 dimensions.");
      }

      // Broadcast shapes for batch dimensions
      std::vector<size_t> batch_shape;
      size_t ndim = std::max(shape_.size(), other.shape_.size()) - 2;
      for (size_t i = 0; i < ndim; ++i) {
        size_t dim_a = (i < shape_.size() - 2) ? shape_[i] : 1;
        size_t dim_b = (i < other.shape_.size() - 2) ? other.shape_[i] : 1;
        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
          throw std::invalid_argument("Batch dimensions are incompatible for matmul.");
        }
        batch_shape.push_back(std::max(dim_a, dim_b));
      }

      size_t M = shape_[shape_.size() - 2];
      size_t K1 = shape_[shape_.size() - 1];
      size_t K2 = other.shape_[other.shape().size() - 2];
      size_t N = other.shape_[other.shape().size() - 1];

      if (K1 != K2) {
        throw std::invalid_argument("Inner tensor dimensions must agree for matmul.");
      }

      // Compute output shape
      std::vector<size_t> result_shape = batch_shape;
      result_shape.push_back(M);
      result_shape.push_back(N);
      Tensor result(result_shape);

      // Flatten batch dimensions
      size_t batch_size = result.size() / (M * N);
      size_t size_a = size_ / batch_size;
      size_t size_b = other.size() / batch_size;

      // Pointers to data
      const double* A = data_.data();
      const double* B = other.data_.data();
      double* C = result.data_.data();

      // Perform batched matrix multiplication
      for (size_t batch = 0; batch < batch_size; ++batch) {
        const double* A_batch = A + batch * size_a;
        const double* B_batch = B + batch * size_b;
        double* C_batch = C + batch * M * N;

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    static_cast<int>(M), static_cast<int>(N), static_cast<int>(K1),
                    1.0, A_batch, static_cast<int>(K1), B_batch, static_cast<int>(N),
                    0.0, C_batch, static_cast<int>(N));
      }
      return result;
    }

private:
    std::vector<size_t> shape_;
    size_t size_ = 0;
    std::vector<double> data_;

    // Calculate total size from shape
    [[nodiscard]] size_t calculateSize(const std::vector<size_t>& shape) const {
      return std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<>());
    }

    // Compute flat index from multi-dimensional indices
    [[nodiscard]] size_t computeFlatIndex(std::initializer_list<size_t> indices) const {
      if (indices.size() != shape_.size()) {
        throw std::out_of_range("Number of indices must match the tensor's dimensions.");
      }
      size_t flat_index = 0;
      size_t stride = 1;
      auto shape_it = shape_.rbegin();
      auto index_it = indices.end();
      do {
        --index_it;
        if (*index_it >= *shape_it) {
          throw std::out_of_range("Index out of bounds.");
        }
        flat_index += (*index_it) * stride;
        stride *= *shape_it;
        ++shape_it;
      } while (shape_it != shape_.rend());
      return flat_index;
    }

    // Check if shapes are compatible for element-wise operations
    void checkShapeCompatibility(const Tensor& other) const {
      if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensor shapes must be identical for this operation.");
      }
    }
};

#endif // QUANT_TRADING_DNN_TENSOR_H
