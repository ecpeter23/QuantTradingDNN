#include "../include/data_loader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <cmath>

// Private member variables for preprocessing
struct NormalizationParams {
    std::vector<double> mean;
    std::vector<double> std_dev;
};

class DataLoaderImpl {
public:
    DataLoaderImpl(const std::string& filepath)
            : filepath_(filepath), train_size_(0), test_size_(0) {}

    void loadData() {
      std::ifstream file(filepath_);
      if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filepath_);
      }

      std::string line;
      // Read the header line to determine the number of columns
      if (!std::getline(file, line)) {
        throw std::runtime_error("Empty file or unable to read header: " + filepath_);
      }

      // Determine the number of features (excluding label)
      std::stringstream ss_header(line);
      std::string item;
      while (std::getline(ss_header, item, ',')) {
        header_.push_back(item);
      }
      if (header_.empty()) {
        throw std::runtime_error("Header is empty in file: " + filepath_);
      }
      num_features_ = header_.size() - 1; // Assuming last column is label

      // Read each data line
      while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> features;
        double value;
        int label;
        int col = 0;
        while (ss >> value) {
          features.push_back(value);
          // Skip comma
          if (ss.peek() == ',') {
            ss.ignore();
          }
          col++;
          if (col == num_features_) {
            break;
          }
        }
        // Read label
        if (!(ss >> label)) {
          throw std::runtime_error("Failed to read label in line: " + line);
        }
        features_.push_back(features);
        labels_.push_back(label);
      }

      file.close();

      if (features_.empty()) {
        throw std::runtime_error("No data found in file: " + filepath_);
      }

      // Preprocess data (e.g., normalization)
      preprocess();
    }

    void preprocess() {
      // Calculate mean and standard deviation for each feature
      NormalizationParams norm_params;
      norm_params.mean.resize(num_features_, 0.0);
      norm_params.std_dev.resize(num_features_, 0.0);

      // Compute mean
      for (const auto& sample : features_) {
        for (size_t i = 0; i < num_features_; ++i) {
          norm_params.mean[i] += sample[i];
        }
      }
      for (size_t i = 0; i < num_features_; ++i) {
        norm_params.mean[i] /= features_.size();
      }

      // Compute standard deviation
      for (const auto& sample : features_) {
        for (size_t i = 0; i < num_features_; ++i) {
          norm_params.std_dev[i] += std::pow(sample[i] - norm_params.mean[i], 2);
        }
      }
      for (size_t i = 0; i < num_features_; ++i) {
        norm_params.std_dev[i] = std::sqrt(norm_params.std_dev[i] / features_.size());
        // Prevent division by zero
        if (norm_params.std_dev[i] == 0.0) {
          norm_params.std_dev[i] = 1.0;
        }
      }

      // Normalize features
      for (auto& sample : features_) {
        for (size_t i = 0; i < num_features_; ++i) {
          sample[i] = (sample[i] - norm_params.mean[i]) / norm_params.std_dev[i];
        }
      }

      // Store normalization parameters if needed for future preprocessing
      norm_params_ = norm_params;
    }

    // Split data into training and test sets (80% train, 20% test)
    void splitData(double train_ratio = 0.8) {
      size_t total_samples = features_.size();
      train_size_ = static_cast<size_t>(total_samples * train_ratio);
      test_size_ = total_samples - train_size_;

      // For time-series data, ensure that training data is earlier than test data
      // No shuffling to prevent look-ahead bias
      train_features_.assign(features_.begin(), features_.begin() + train_size_);
      train_labels_.assign(labels_.begin(), labels_.begin() + train_size_);

      test_features_.assign(features_.begin() + train_size_, features_.end());
      test_labels_.assign(labels_.begin() + train_size_, labels_.end());
    }

    // Getters for training and test data
    std::pair<std::vector<std::vector<double>>, std::vector<int>> getTrainingData() const {
      return { train_features_, train_labels_ };
    }

    std::pair<std::vector<std::vector<double>>, std::vector<int>> getTestData() const {
      return { test_features_, test_labels_ };
    }

private:
    std::string filepath_;
    std::vector<std::string> header_;
    size_t num_features_;
    std::vector<std::vector<double>> features_;
    std::vector<int> labels_;

    // Normalization parameters
    NormalizationParams norm_params_;

    // Split data
    size_t train_size_;
    size_t test_size_;
    std::vector<std::vector<double>> train_features_;
    std::vector<int> train_labels_;
    std::vector<std::vector<double>> test_features_;
    std::vector<int> test_labels_;
};

DataLoader::DataLoader(std::string filepath)
        : filepath_(std::move(filepath)) {
  // Initialization if needed
}

std::pair<std::vector<std::vector<double>>, std::vector<int>> DataLoader::loadTrainingData() {
  DataLoaderImpl loader(filepath_);
  loader.loadData();
  loader.splitData(0.8); // 80% training, 20% testing
  return loader.getTrainingData();
}

std::pair<std::vector<std::vector<double>>, std::vector<int>> DataLoader::loadTestData() {
  DataLoaderImpl loader(filepath_);
  loader.loadData();
  loader.splitData(0.8); // 80% training, 20% testing
  return loader.getTestData();
}


