#include "data_loader.h"
#include "neural_network.h"
#include "layers/fully_connected.h"
#include "layers/activation.h"
#include <iostream>
#include <filesystem>

int main() {
  try {
    // Define the path to the CSV file containing historical trading data
    std::string data_filepath = "/Users/elipeter/CLionProjects/quant-trading-dnn/data/historical_prices.csv";

    // Initialize DataLoader with the specified CSV file path
    DataLoader dataLoader(data_filepath);
    std::cout << "DataLoader initialized with file: " << data_filepath << std::endl;

    // Load and preprocess training data
    auto [trainData, trainLabels] = dataLoader.loadTrainingData();
    std::cout << "Loaded Training Data: " << trainData.size() << " samples." << std::endl;

    // Load and preprocess test data
    auto [testData, testLabels] = dataLoader.loadTestData();
    std::cout << "Loaded Test Data: " << testData.size() << " samples." << std::endl;

    // Validate that the loaded data is not empty
    if (trainData.empty() || trainLabels.empty()) {
      std::cerr << "Error: Training data or labels are empty." << std::endl;
      return EXIT_FAILURE;
    }
    if (testData.empty() || testLabels.empty()) {
      std::cerr << "Error: Test data or labels are empty." << std::endl;
      return EXIT_FAILURE;
    }

    // Determine the number of input features from the training data
    size_t input_size = trainData[0].size();
    std::cout << "Number of Input Features: " << input_size << std::endl;

    // Define hyperparameters
    size_t hidden_size = 16;     // Number of neurons in the hidden layer
    size_t output_size = 1;      // Output size for binary classification
    int epochs = 100;            // Number of training epochs
    double learningRate = 0.01;  // Learning rate for weight updates

    // Initialize the Neural Network
    NeuralNetwork nn;
    nn.addLayer<FullyConnected>(input_size, hidden_size);
    nn.addLayer<Activation>(ActivationType::ReLU);
    nn.addLayer<FullyConnected>(hidden_size, output_size);
    nn.addLayer<Activation>(ActivationType::Sigmoid);
    std::cout << "Neural Network architecture initialized." << std::endl;

    // Train the Neural Network
    std::cout << "Starting training for " << epochs << " epochs with learning rate " << learningRate << "..." << std::endl;
    nn.train(trainData, trainLabels, epochs, learningRate);
    std::cout << "Training completed." << std::endl;

    // Evaluate the Neural Network on test data
    std::cout << "Evaluating the model on test data..." << std::endl;
    double accuracy = nn.evaluate(testData, testLabels);
    std::cout << "Test Accuracy: " << accuracy << "%" << std::endl;

    // Generate predictions on test data
    std::vector<int> predictions = nn.predict(testData);

    // Define the path to save the predictions
    std::string predictions_dir = "/Users/elipeter/CLionProjects/quant-trading-dnn/predictions";
    std::string predictions_file = "/Users/elipeter/CLionProjects/quant-trading-dnn/predictions/test_predictions.csv";

    // Ensure the predictions directory exists
    if (!std::filesystem::exists(predictions_dir)) {
      if (std::filesystem::create_directory(predictions_dir)) {
        std::cout << "Created directory: predictions" << std::endl;
      } else {
        std::cerr << "Error: Failed to create directory: predictions" << std::endl;
        return EXIT_FAILURE;
      }
    }

    // Save predictions to CSV
    std::ofstream ofs(predictions_file);
    if(!ofs.is_open()) {
      std::cerr << "Failed to open file for saving predictions: " << predictions_file << std::endl;
      return EXIT_FAILURE;
    }

    ofs << "Prediction\n";
    for(const auto& pred : predictions) {
      ofs << pred << "\n";
    }

    ofs.close();
    std::cout << "Predictions saved to " << predictions_file << std::endl;

    // Optionally, save the true labels for comparison
    std::string true_labels_file = "/Users/elipeter/CLionProjects/quant-trading-dnn/predictions/test_labels.csv";
    std::ofstream ofs_labels(true_labels_file);
    if(!ofs_labels.is_open()) {
      std::cerr << "Failed to open file for saving true labels: " << true_labels_file << std::endl;
      return EXIT_FAILURE;
    }

    ofs_labels << "TrueLabel\n";
    for(const auto& label : testLabels) {
      ofs_labels << label << "\n";
    }

    ofs_labels.close();
    std::cout << "True labels saved to " << true_labels_file << std::endl;

    // Save the trained model to the specified path
    std::string model_directory = "/Users/elipeter/CLionProjects/quant-trading-dnn/models";
    std::string model_filename = "trained_model.bin";
    std::string model_path = model_directory + "/" + model_filename;

    // Ensure the model directory exists; create it if it doesn't
    if (!std::filesystem::exists(model_directory)) {
      if (std::filesystem::create_directory(model_directory)) {
        std::cout << "Created directory: " << model_directory << std::endl;
      } else {
        std::cerr << "Error: Failed to create directory: " << model_directory << std::endl;
        return EXIT_FAILURE;
      }
    }

    // Save the trained model to the specified path
    nn.saveModel(model_path);
    std::cout << "Trained model saved to: " << model_path << std::endl;

  } catch (const std::exception& e) {
    // Catch and display any exceptions that occur during execution
    std::cerr << "An exception occurred: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
