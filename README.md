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
#include "Network.h"
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

    // Initialize the network with a topology of 3 input nodes, 3 hidden nodes, and 1 output node
    Network<double, ActivationType::Sigmoid> network({3, 3, 1});

    // Training parameters
    double learni
