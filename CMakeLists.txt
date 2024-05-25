# NeuralNet

A C++ library for building and training simple neural networks. This project uses template metaprogramming to support various activation functions, loss functions, and optimization algorithms. The primary focus is on creating a lean and mean neural network framework that can be easily extended and customized.

## Features

- **Forward Propagation**: Implemented for multi-layer perceptrons.
- **Backward Propagation**: Basic structure implemented for updating weights and biases.
- **Activation Functions**: Supports Sigmoid, ReLU, and easily extendable for others.
- **Loss Functions**: Supports Mean Squared Error (MSE) and Cross-Entropy.
- **Template Metaprogramming**: Allows easy extension for different types of activation functions, loss functions, and optimizers.
- **Layer and Node Abstraction**: Clean separation of layers and nodes for better organization and maintainability.
- **Future OpenML Support**: Planned integration with OpenML for more advanced machine learning tasks.

## Getting Started

### Prerequisites

- C++17 or later
- CMake
- GCC or Clang
- GoogleTest (for running unit tests)

### Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/NeuralNet.git
    cd NeuralNet
    ```

2. **Build the project:**

    ```bash
    mkdir build
    cd build
    cmake ..
    make
    ```

3. **Run the tests:**

    ```bash
    ./NeuralNetTests
    ```

## Usage

### Basic Example

Here's a simple example to create and train a neural network for a 3-input AND gate:

```cpp
#include <iostream>
#include <vector>
#include "Model.h"
#include "Utility.h"
#include "LossFunction.h"

int main() {
    // Define the AND Gate training data for 3 inputs
    std::vector<std::vector<double>> andInputs = {
        {0, 0, 0},
        {0, 0, 1},
        {0, 1, 0},
        {0, 1, 1},
        {1, 0, 0},
        {1, 0, 1},
        {1, 1, 0},
        {1, 1, 1}
    };

    std::vector<std::vector<double>> andOutputs = {
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {0},
        {1}
    };

    // Initialize the model with a topology of 3 input nodes, 3 hidden nodes, and 1 output node
    Model<double, ActivationType::Sigmoid> model({3, 3, 1});

    // Training parameters
    double learningRate = 0.1;
    int epochs = 10000;

    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double totalLoss = 0.0;
        for (size_t i = 0; i < andInputs.size(); ++i) {
            std::vector<double> output = model.forward(andInputs[i]);
            totalLoss += LossFunction::meanSquaredError(output, andOutputs[i]);
            model.backward(LossFunction::meanSquaredErrorDerivative(output, andOutputs[i]));
            model.updateWeights(learningRate);
        }
        if (epoch % 1000 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << totalLoss / andInputs.size() << std::endl;
        }
    }

    // Test the trained model
    for (const auto& input : andInputs) {
        std::vector<double> output = model.forward(input);
        std::cout << "Input: ";
        for (double val : input) {
            std::cout << val << " ";
        }
        std::cout << "Output: " << output[0] << std::endl;
    }

    return 0;
}
