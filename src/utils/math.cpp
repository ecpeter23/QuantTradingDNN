#include "../../include/utils/math.h"
#include <Accelerate/Accelerate.h>
#include <cassert>

std::vector<double> matMul(const std::vector<std::vector<double>>& matrix, const std::vector<double>& vec) {
  size_t rows = matrix.size();
  size_t cols = matrix[0].size();
  assert(vec.size() == cols && "Matrix columns must match vector size");

  std::vector<double> result(rows, 0.0);

  // Flatten the matrix for Accelerate
  std::vector<double> flatMatrix;
  flatMatrix.reserve(rows * cols);
  for(const auto& row : matrix) {
    flatMatrix.insert(flatMatrix.end(), row.begin(), row.end());
  }

  // Perform matrix-vector multiplication: y = A * x
  // cblas_dgemv expects column-major order by default, but we're using row-major
  // So we set CblasRowMajor
  cblas_dgemv(CblasRowMajor, CblasNoTrans, static_cast<int>(rows), static_cast<int>(cols),
              1.0, flatMatrix.data(), static_cast<int>(cols),
              vec.data(), 1, 0.0, result.data(), 1);

  return result;
}
