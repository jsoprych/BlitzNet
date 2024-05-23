#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <stdexcept>
#include "Layer.h"

template <typename T, ActivationType Activation>
class Network {
public:
    Network(const std::initializer_list<int>& layerSizes) {
        auto it = layerSizes.begin();
        int numInputs = *it; // First element is the number of inputs
        ++it;
        addInputLayer(numInputs); // Special handling for input layer
        for (; it != layerSizes.end(); ++it) {
            addLayer(*it, numInputs);
            numInputs = *it;
        }
    }

    void addInputLayer(int numNodes) {
        layers.emplace_back(numNodes, 1, true); // Each input node has 1 input
    }

    void addLayer(int numNodes, int numInputsPerNode) {
        layers.emplace_back(numNodes, numInputsPerNode, false); // Pass numInputsPerNode to the layer constructor
    }

    std::vector<T> forward(const std::vector<T>& inputs) const {
        std::vector<T> currentInputs = inputs;
        for (const auto& layer : layers) {
            currentInputs = layer.forward(currentInputs);
        }
        return currentInputs;
    }

    const std::vector<Layer<T, Activation>>& getLayers() const {
        return layers;
    }

private:
    std::vector<Layer<T, Activation>> layers;
};

#endif // NETWORK_H
