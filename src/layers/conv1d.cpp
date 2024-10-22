//
// Created by Eli Peter on 10/21/24.
//

#include "layers/conv1d.h"
#include <cmath>
#include <iostream>
#include <random>
#include <Accelerate/Accelerate.h>

// Constructor: Initializes weights, biases, and gradients
Conv1D::Conv1D(size_t input_channels, size_t output_channels, size_t kernel_size, size_t stride, size_t padding)
        : input_channels_(input_channels), output_channels_(output_channels), kernel_size_(kernel_size),
          stride_(stride), padding_(padding) {
  // Initialize weights and biases with small random values
  std::mt19937 gen(std::random_device{}());
  std::uniform_real_distribution<> dist(-0.1, 0.1);

  weights_.resize(output_channels_, std::vector<std::vector<double>>(input_channels_, std::vector<double>(kernel_size_, 0.0)));
  biases_.resize(output_channels_, 0.0);

  // Initialize weights
  for(auto& oc : weights_) {
    for(auto& ic : oc) {
      for(auto& w : ic) {
        w = dist(gen);
      }
    }
  }

  // Initialize biases
  for(auto& b : biases_) {
    b = dist(gen);
  }

  // Initialize gradients
  grad_weights_.resize(output_channels_, std::vector<std::vector<double>>(input_channels_, std::vector<double>(kernel_size_, 0.0)));
  grad_biases_.resize(output_channels_, 0.0);
}

// Utility function to pad the input
void Conv1D::padInput(const std::vector<double>& input) {
  // Assuming input is flattened as [input_channels][input_length]
  // Determine input_length from input vector size and input_channels
  input_length_ = input.size() / input_channels_;

  // Initialize padded input channels
  input_channels_padded_.resize(input_channels_, std::vector<double>(input_length_ + 2 * padding_, 0.0));

  for(size_t c = 0; c < input_channels_; ++c) {
    for(size_t l = 0; l < input_length_; ++l) {
      input_channels_padded_[c][l + padding_] = input[c * input_length_ + l];
    }
  }
}

// Forward pass: Performs convolution operation
std::vector<double> Conv1D::forward(const std::vector<double>& input) {
  // Pad the input
  padInput(input);

  // Calculate output length
  size_t output_length = (input_length_ + 2 * padding_ - kernel_size_) / stride_ + 1;

  // Initialize output channels
  std::vector<std::vector<double>> output_channels(output_channels_, std::vector<double>(output_length, 0.0));

  // Perform convolution for each output channel
  for(size_t oc = 0; oc < output_channels_; ++oc) {
    for(size_t oc_in = 0; oc_in < input_channels_; ++oc_in) {
      // Get the filter for this output and input channel
      const std::vector<double>& filter = weights_[oc][oc_in];

      // Perform convolution using vDSP_conv for this input channel
      // vDSP_conv performs convolution of a filter over an input signal

      // Input signal for this channel
      const double* input_ptr = input_channels_padded_[oc_in].data();
      size_t input_padded_length = input_length_ + 2 * padding_;

      // Output signal pointer
      double* output_ptr = output_channels[oc].data();

      // Temporary buffer to store convolution result
      // Output length as calculated
      size_t conv_length = output_length;

      // Perform convolution
      // vDSP_conv requires the filter to be in the correct format and the output buffer
      vDSP_convD(input_ptr, 1, filter.data(), 1, output_ptr, 1, conv_length, kernel_size_);
    }

    // Add bias
    for(size_t l = 0; l < output_length; ++l) {
      output_channels[oc][l] += biases_[oc];
    }
  }

  // Flatten the output channels into a single output vector
  std::vector<double> output;
  output.reserve(output_channels_ * output_length);
  for(const auto& oc_output : output_channels) {
    output.insert(output.end(), oc_output.begin(), oc_output.end());
  }

  return output;
}

// Backward pass: Computes gradients with respect to inputs and weights
std::vector<double> Conv1D::backward(const std::vector<double>& gradOutput) {
  // Calculate output length
  size_t output_length = (input_length_ + 2 * padding_ - kernel_size_) / stride_ + 1;

  // Reshape gradOutput to [output_channels][output_length]
  std::vector<std::vector<double>> gradOutput_channels(output_channels_, std::vector<double>(output_length, 0.0));
  for(size_t oc = 0; oc < output_channels_; ++oc) {
    for(size_t l = 0; l < output_length; ++l) {
      gradOutput_channels[oc][l] = gradOutput[oc * output_length + l];
    }
  }

  // Initialize gradients w.r.t input
  std::vector<std::vector<double>> grad_input_padded(input_channels_, std::vector<double>(input_length_ + 2 * padding_, 0.0));

  // Compute gradients w.r.t weights and biases
  for(size_t oc = 0; oc < output_channels_; ++oc) {
    for(size_t oc_in = 0; oc_in < input_channels_; ++oc_in) {
      // Get the filter
      const std::vector<double>& filter = weights_[oc][oc_in];

      // Get the input signal
      const std::vector<double>& input_padded = input_channels_padded_[oc_in];

      // Get the gradOutput signal for this output channel
      const std::vector<double>& gradOutput_signal = gradOutput_channels[oc];

      // Perform convolution of input with gradOutput to get grad_weights
      // For grad_weights, the gradient is input convolved with gradOutput

      // Initialize a temporary buffer to store the gradient for this filter
      std::vector<double> grad_filter(kernel_size_, 0.0);

      // Perform correlation (since vDSP_conv performs convolution, which includes flipping the filter)
      // To compute the gradient with respect to the filter, perform correlation
      for(size_t l = 0; l < output_length; ++l) {
        for(size_t k = 0; k < kernel_size_; ++k) {
          size_t input_idx = l * stride_ + k;
          if(input_idx < input_length_ + 2 * padding_) {
            grad_filter[k] += input_padded[input_idx] * gradOutput_signal[l];
          }
        }
      }

      // Accumulate gradients
      for(size_t k = 0; k < kernel_size_; ++k) {
        grad_weights_[oc][oc_in][k] += grad_filter[k];
      }

      // Compute gradient w.r.t bias
      double grad_bias = 0.0;
      for(size_t l = 0; l < output_length; ++l) {
        grad_bias += gradOutput_signal[l];
      }
      grad_biases_[oc] += grad_bias;

      // Compute gradient w.r.t input_padded
      // To compute grad_input, convolve gradOutput with the flipped filter
      // Since vDSP_conv flips the filter, to perform the correct operation, we need to reverse the filter

      // Reverse the filter for convolution
      std::vector<double> reversed_filter(kernel_size_, 0.0);
      for(size_t k = 0; k < kernel_size_; ++k) {
        reversed_filter[k] = filter[kernel_size_ - 1 - k];
      }

      // Perform convolution using vDSP_conv
      const double* gradOutput_ptr = gradOutput_signal.data();
      const double* reversed_filter_ptr = reversed_filter.data();
      size_t grad_conv_length = input_length_ + 2 * padding_ - kernel_size_ + 1;

      // Temporary buffer to store convolution result
      std::vector<double> grad_conv_result(grad_conv_length, 0.0);

      vDSP_convD(gradOutput_ptr, 1, reversed_filter_ptr, 1, grad_conv_result.data(), 1, grad_conv_length, kernel_size_);

      // Accumulate gradients w.r.t input_padded
      for(size_t l = 0; l < grad_conv_length; ++l) {
        size_t input_idx = l;
        if(input_idx < input_length_ + 2 * padding_) {
          grad_input_padded[oc_in][input_idx] += grad_conv_result[l];
        }
      }
    }
  }

  // Remove padding from grad_input_padded to get grad_input
  std::vector<double> grad_input(input_channels_ * input_length_, 0.0);
  for(size_t oc_in = 0; oc_in < input_channels_; ++oc_in) {
    for(size_t l = 0; l < input_length_; ++l) {
      grad_input[oc_in * input_length_ + l] = grad_input_padded[oc_in][l + padding_];
    }
  }

  return grad_input;
}

// Update weights and biases using gradients
void Conv1D::updateWeights(double learningRate) {
  // Update weights
  for(size_t oc = 0; oc < output_channels_; ++oc) {
    for(size_t oc_in = 0; oc_in < input_channels_; ++oc_in) {
      for(size_t k = 0; k < kernel_size_; ++k) {
        weights_[oc][oc_in][k] -= learningRate * grad_weights_[oc][oc_in][k];
        // Reset gradient after update
        grad_weights_[oc][oc_in][k] = 0.0;
      }
    }
  }

  // Update biases
  for(size_t oc = 0; oc < output_channels_; ++oc) {
    biases_[oc] -= learningRate * grad_biases_[oc];
    // Reset gradient after update
    grad_biases_[oc] = 0.0;
  }
}

// Save the layer's parameters to a binary file
void Conv1D::save(std::ofstream& ofs) const {
  // Save layer type identifier
  std::string layer_type = "Conv1D";
  size_t type_length = layer_type.size();
  ofs.write(reinterpret_cast<const char*>(&type_length), sizeof(size_t));
  ofs.write(layer_type.c_str(), type_length);

  // Save parameters
  ofs.write(reinterpret_cast<const char*>(&input_channels_), sizeof(size_t));
  ofs.write(reinterpret_cast<const char*>(&output_channels_), sizeof(size_t));
  ofs.write(reinterpret_cast<const char*>(&kernel_size_), sizeof(size_t));
  ofs.write(reinterpret_cast<const char*>(&stride_), sizeof(size_t));
  ofs.write(reinterpret_cast<const char*>(&padding_), sizeof(size_t));

  // Save weights and biases
  for(const auto& oc : weights_) {
    for(const auto& ic : oc) {
      ofs.write(reinterpret_cast<const char*>(ic.data()), ic.size() * sizeof(double));
    }
  }

  for(const auto& b : biases_) {
    ofs.write(reinterpret_cast<const char*>(&b), sizeof(double));
  }
}

// Load the layer's parameters from a binary file
std::unique_ptr<Layer> Conv1D::load(std::ifstream& ifs) {
  size_t input_channels, output_channels, kernel_size, stride, padding;
  ifs.read(reinterpret_cast<char*>(&input_channels), sizeof(size_t));
  ifs.read(reinterpret_cast<char*>(&output_channels), sizeof(size_t));
  ifs.read(reinterpret_cast<char*>(&kernel_size), sizeof(size_t));
  ifs.read(reinterpret_cast<char*>(&stride), sizeof(size_t));
  ifs.read(reinterpret_cast<char*>(&padding), sizeof(size_t));

  auto layer = std::make_unique<Conv1D>(input_channels, output_channels, kernel_size, stride, padding);

  // Load weights and biases
  for(auto& oc : layer->weights_) {
    for(auto& ic : oc) {
      ifs.read(reinterpret_cast<char*>(ic.data()), ic.size() * sizeof(double));
    }
  }

  for(auto& b : layer->biases_) {
    ifs.read(reinterpret_cast<char*>(&b), sizeof(double));
  }

  return layer;
}
