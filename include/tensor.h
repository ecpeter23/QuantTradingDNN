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

    // Similar implementations for -, *, / with scalar

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

    // Matrix multiplication (for 2D tensors)
    [[nodiscard]] Tensor matmul(const Tensor& other) const {
      if (shape_.size() != 2 || other.shape_.size() != 2) {
        throw std::logic_error("Matrix multiplication is only defined for 2D tensors.");
      }
      if (shape_[1] != other.shape_[0]) {
        throw std::invalid_argument("Inner tensor dimensions must agree.");
      }
      size_t M = shape_[0];              // Rows of the first matrix
      size_t K = shape_[1];              // Columns of the first matrix and rows of the second
      size_t N = other.shape_[1];        // Columns of the second matrix

      Tensor result({M, N});

      // Parameters for cblas_dgemm
      const double* A = data_.data();
      const double* B = other.data_.data();
      double* C = result.data_.data();

      CBLAS_ORDER order = CblasRowMajor;
      CBLAS_TRANSPOSE transA = CblasNoTrans;
      CBLAS_TRANSPOSE transB = CblasNoTrans;

      double alpha = 1.0;
      double beta = 0.0;

      // Perform matrix multiplication: C = alpha * A * B + beta * C
      cblas_dgemm(order, transA, transB, (int)M, (int)N, (int)K, alpha, A, (int)K, B, (int)N, beta, C, (int)N);

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


#endif //QUANT_TRADING_DNN_TENSOR_H
