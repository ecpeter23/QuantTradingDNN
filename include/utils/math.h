//
// Created by Eli Peter on 10/21/24.
//

#ifndef QUANT_TRADING_DNN_MATH_H
#define QUANT_TRADING_DNN_MATH_H

#include <vector>

// Matrix-vector multiplication using Accelerate framework
std::vector<double> matMul(const std::vector<std::vector<double>>& matrix, const std::vector<double>& vec);


#endif //QUANT_TRADING_DNN_MATH_H
