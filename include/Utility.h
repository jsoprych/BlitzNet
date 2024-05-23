#ifndef UTILITY_H
#define UTILITY_H

#include <iostream>
#include "Network.h"

template <typename T, ActivationType Activation>
void printNetworkState(const Network<T, Activation>& network) {
    const auto& layers = network.getLayers();
    for (size_t i = 0; i < layers.size(); ++i) {
        const auto& nodes = layers[i].getNodes();
        std::cout << "Layer " << i << ":\n";
        for (size_t j = 0; j < nodes.size(); ++j) {
            const auto& node = nodes[j];
            std::cout << "  Node " << j << ": Weights: ";
            for (const auto& weight : node.getWeights()) {
                std::cout << weight << " ";
            }
            std::cout << "Bias: " << node.getBias() << " Output: " << node.getOutput() << "\n";
        }
    }
}

#endif // UTILITY_H
