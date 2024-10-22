#include "../include/neural_network.h"
#include "../include/utils/bce_loss.h"
#include <iostream>
#include <fstream>
#include <ranges>

NeuralNetwork::NeuralNetwork() {
  // Initialize network parameters if needed
}

void NeuralNetwork::train(const std::vector<std::vector<double>>& data, const std::vector<int>& labels, int epochs, double learningRate) {
  if(data.empty() || data.size() != labels.size()) {
    std::cerr << "Data and labels must be non-empty and of the same size." << std::endl;
    return;
  }

  size_t num_samples = data.size();

  for(int epoch = 0; epoch < epochs; ++epoch) {
    double epoch_loss = 0.0;

    for(size_t i = 0; i < num_samples; ++i) {
      // Forward Pass
      std::vector<double> activation = data[i];
      for(auto& layer : layers_) {
        activation = layer->forward(activation);
      }

      // Compute Loss (BCE)
      double loss = computeBCE(activation, { labels[i] });
      epoch_loss += loss;

      // Compute Gradient of Loss w.r.t. Output (BCE)
      std::vector<double> grad_output = computeBCEGradient(activation, { labels[i] });

      // Backward Pass
      for(auto & layer : std::ranges::reverse_view(layers_)) {
        grad_output = layer->backward(grad_output);
      }

      // Update Weights
      for(auto& layer : layers_) {
        layer->updateWeights(learningRate);
      }
    }

    // Compute average loss for the epoch
    epoch_loss /= num_samples;
    std::cout << "Epoch " << epoch + 1 << "/" << epochs << " - Loss: " << epoch_loss << std::endl;
  }
}

double NeuralNetwork::evaluate(const std::vector<std::vector<double>>& data, const std::vector<int>& labels) {
  if(data.empty() || data.size() != labels.size()) {
    std::cerr << "Data and labels must be non-empty and of the same size." << std::endl;
    return 0.0;
  }

  size_t num_samples = data.size();
  double total_loss = 0.0;
  size_t correct_predictions = 0;

  for(size_t i = 0; i < num_samples; ++i) {
    // Forward Pass
    std::vector<double> activation = data[i];
    for(auto& layer : layers_) {
      activation = layer->forward(activation);
    }

    // Assuming binary classification with sigmoid activation in the output layer
    double prediction = activation[0] >= 0.5 ? 1.0 : 0.0;
    if(static_cast<int>(prediction) == labels[i]) {
      correct_predictions += 1;
    }

    // Compute Loss (MSE)
    double error = activation[0] - labels[i];
    total_loss += error * error;
  }

  double average_loss = total_loss / num_samples;
  double accuracy = (static_cast<double>(correct_predictions) / num_samples) * 100.0;

  std::cout << "Evaluation - Loss: " << average_loss << ", Accuracy: " << accuracy << "%" << std::endl;

  return accuracy;
}


void NeuralNetwork::saveModel(const std::string& filepath) {
  std::ofstream ofs(filepath, std::ios::binary);
  if(!ofs.is_open()) {
    std::cerr << "Failed to open file for saving model: " << filepath << std::endl;
    return;
  }

  // Save the number of layers
  size_t num_layers = layers_.size();
  ofs.write(reinterpret_cast<char*>(&num_layers), sizeof(size_t));

  // Save each layer
  for(auto& layer : layers_) {
    layer->save(ofs);
  }

  ofs.close();
  std::cout << "Model saved to " << filepath << std::endl;
}

void NeuralNetwork::loadModel(const std::string& filepath) {
  std::ifstream ifs(filepath, std::ios::binary);
  if(!ifs.is_open()) {
    std::cerr << "Failed to open file for loading model: " << filepath << std::endl;
    return;
  }

  // Clear existing layers
  layers_.clear();

  // Read the number of layers
  size_t num_layers;
  ifs.read(reinterpret_cast<char*>(&num_layers), sizeof(size_t));

  // Load each layer
  for(size_t i = 0; i < num_layers; ++i) {
    // Assuming each layer has a static load method that returns a unique_ptr to a Layer
    // You might need to implement a factory method to handle different layer types
    std::unique_ptr<Layer> layer = Layer::load(ifs);
    if(layer) {
      layers_.emplace_back(std::move(layer));
    } else {
      std::cerr << "Failed to load layer " << i << std::endl;
      break;
    }
  }

  ifs.close();
  std::cout << "Model loaded from " << filepath << std::endl;
}

std::vector<int> NeuralNetwork::predict(const std::vector<std::vector<double>>& data) {
  std::vector<int> predictions;
  for(const auto& sample : data) {
    std::vector<double> activation = sample;
    for(auto& layer : layers_) {
      activation = layer->forward(activation);
    }
    double pred = activation[0] >= 0.5 ? 1 : 0;
    predictions.push_back(static_cast<int>(pred));
  }
  return predictions;
}
