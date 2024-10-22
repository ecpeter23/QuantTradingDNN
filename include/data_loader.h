#ifndef QUANT_TRADING_DNN_DATA_LOADER_H
#define QUANT_TRADING_DNN_DATA_LOADER_H

#include <vector>
#include <string>

class DataLoader {
public:
    explicit DataLoader(std::string  filepath);
    std::pair<std::vector<std::vector<double>>, std::vector<int>> loadTrainingData();
    std::pair<std::vector<std::vector<double>>, std::vector<int>> loadTestData();
protected:
    std::string filepath_;
    void preprocess();
    // Add any necessary member variables
};

#endif //QUANT_TRADING_DNN_DATA_LOADER_H
