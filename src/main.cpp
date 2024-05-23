#include <iostream>
#include "Network.h"
#include "Utility.h"

int main() {
    // Initialize network for AND gate with 3 inputs, 1 hidden layer with 3 neurons, and 1 output neuron
    Network<double, ActivationType::Sigmoid> network({3, 5, 1});
    
    // Define AND gate inputs and expected outputs
    std::vector<std::vector<double>> and_inputs = {
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 1.0},
        {0.0, 1.0, 0.0},
        {0.0, 1.0, 1.0},
        {1.0, 0.0, 0.0},
        {1.0, 0.0, 1.0},
        {1.0, 1.0, 0.0},
        {1.0, 1.0, 1.0}
    };
    std::vector<double> and_outputs = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0};

    // Perform forward pass for each input set
    for (size_t i = 0; i < and_inputs.size(); ++i) {
        std::vector<double> output = network.forward(and_inputs[i]);
        std::cout << "Input: " << and_inputs[i][0] << ", " << and_inputs[i][1] << ", " << and_inputs[i][2]
                  << " Output: " << output[0] << " Expected: " << and_outputs[i] << std::endl;
    }
    
    // Print network state
    printNetworkState(network);

    return 0;
}
