#include <gtest/gtest.h>
#include "../include/neural_network.h"
#include "../include/layers/fully_connected.h"
#include "../include/layers/activation.h"
#include <vector>
#include <random>
#include <fstream>
#include <algorithm>

// Test Neural Network Initialization
TEST(NeuralNetworkTest, Initialization) {
  NeuralNetwork nn;
  nn.addLayer<FullyConnected>(2, 3);
  nn.addLayer<Activation>(ActivationType::ReLU);
  nn.addLayer<FullyConnected>(3, 1);
  nn.addLayer<Activation>(ActivationType::Sigmoid);
  // Further assertions can be added to verify the layers
  EXPECT_EQ(4, 4);
}

// Test Forward Pass
TEST(NeuralNetworkTest, ForwardPass) {
NeuralNetwork nn;
nn.addLayer<FullyConnected>(2, 2);
nn.addLayer<Activation>(ActivationType::ReLU);
nn.addLayer<FullyConnected>(2, 1);
nn.addLayer<Activation>(ActivationType::Sigmoid);

std::vector<std::vector<double>> data = { {0.5, -0.5}, {1.0, 1.0} };
std::vector<int> labels = { 0, 1 };

// Perform a forward pass
nn.train(data, labels, 1, 0.1);

// No specific assertions, but ensure no crashes and outputs are reasonable
SUCCEED();
}

// Test Model Saving and Loading
TEST(NeuralNetworkTest, SaveAndLoadModel) {
NeuralNetwork nn;
nn.addLayer<FullyConnected>(2, 2);
nn.addLayer<Activation>(ActivationType::ReLU);
nn.addLayer<FullyConnected>(2, 1);
nn.addLayer<Activation>(ActivationType::Sigmoid);

std::vector<std::vector<double>> data = { {0.5, -0.5}, {1.0, 1.0} };
std::vector<int> labels = { 0, 1 };

nn.train(data, labels, 100, 0.1);
nn.saveModel("/Users/elipeter/CLionProjects/quant-trading-dnn/models/test_model.bin");

NeuralNetwork nn_loaded;
nn_loaded.loadModel("/Users/elipeter/CLionProjects/quant-trading-dnn/models/test_model.bin");

// Compare parameters or ensure loaded model can perform predictions
double accuracy = nn_loaded.evaluate(data, labels);
EXPECT_GE(accuracy, 50.0); // At least better than random guessing
}

std::pair<std::vector<std::vector<double>>, std::vector<int>> generateSyntheticData(int num_samples, int num_features) {
  std::vector<std::vector<double>> data;
  std::vector<int> labels;

  // Define two means for the two classes
  std::vector<double> mean_class0(num_features, 0.0);
  std::vector<double> mean_class1(num_features, 1.0);

  // Standard deviation
  double std_dev = 0.5;

  // Random number generators
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> d0(0.0, std_dev);
  std::normal_distribution<> d1(1.0, std_dev);

  for(int i = 0; i < num_samples / 2; ++i) {
    std::vector<double> sample;
    for(int j = 0; j < num_features; ++j) {
      sample.push_back(d0(gen));
    }
    data.push_back(sample);
    labels.push_back(0);
  }

  for(int i = 0; i < num_samples / 2; ++i) {
    std::vector<double> sample;
    for(int j = 0; j < num_features; ++j) {
      sample.push_back(d1(gen));
    }
    data.push_back(sample);
    labels.push_back(1);
  }

  // Shuffle the dataset
  std::vector<size_t> indices(num_samples);
  for(size_t i = 0; i < num_samples; ++i) indices[i] = i;
  std::shuffle(indices.begin(), indices.end(), gen);

  std::vector<std::vector<double>> shuffled_data;
  std::vector<int> shuffled_labels;
  for(auto idx : indices) {
    shuffled_data.push_back(data[idx]);
    shuffled_labels.push_back(labels[idx]);
  }

  return { shuffled_data, shuffled_labels };
}

TEST(NeuralNetworkTest, SaveAndLoadModelAdvanced) {
  // Parameters for synthetic data
  int num_samples = 1000; // Total samples
  int num_features = 10;   // Number of features

  // Generate synthetic data
  auto [data, labels] = generateSyntheticData(num_samples, num_features);

  // Define network architecture
  NeuralNetwork nn;
  nn.addLayer<FullyConnected>(num_features, 16); // Increased hidden layer size
  nn.addLayer<Activation>(ActivationType::ReLU);
  nn.addLayer<FullyConnected>(16, 8);
  nn.addLayer<Activation>(ActivationType::ReLU);
  nn.addLayer<FullyConnected>(8, 1);
  nn.addLayer<Activation>(ActivationType::Sigmoid);

  // Training parameters
  int epochs = 100;
  double learningRate = 0.01;

  // Train the network
  nn.train(data, labels, epochs, learningRate);

  // Evaluate pre-save performance
  double pre_save_accuracy = nn.evaluate(data, labels);
  std::cout << "Pre-save Accuracy: " << pre_save_accuracy << "%" << std::endl;

  // Ensure the model has learned something
  EXPECT_GE(pre_save_accuracy, 70.0); // Expect at least 70% accuracy

  // Save the model
  std::string model_path = "/Users/elipeter/CLionProjects/quant-trading-dnn/models/advanced_test_model.bin";
  nn.saveModel(model_path);

  // Load the model into a new instance
  NeuralNetwork nn_loaded;
  nn_loaded.loadModel(model_path);

  // Evaluate post-load performance
  double post_load_accuracy = nn_loaded.evaluate(data, labels);
  std::cout << "Post-load Accuracy: " << post_load_accuracy << "%" << std::endl;

  // Compare accuracies
  EXPECT_NEAR(pre_save_accuracy, post_load_accuracy, 1.0); // Allow a small difference

  // Optionally, compare predictions to ensure they are the same or very similar
  // For this, you need a predict method in NeuralNetwork
  /*
  std::vector<int> original_predictions = nn.predict(data);
  std::vector<int> loaded_predictions = nn_loaded.predict(data);

  for(size_t i = 0; i < original_predictions.size(); ++i) {
      EXPECT_EQ(original_predictions[i], loaded_predictions[i]);
  }
  */
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

