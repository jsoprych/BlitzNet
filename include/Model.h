#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <stdexcept>
#include "Layer.h"

template <typename T, ActivationType Activation>
class Model {
public:
    Model(const std::initializer_list<int>& layerSizes) {
        auto it = layerSizes.begin();
        int numInputs = *it; // First element is the number of inputs
        ++it;
        addInputLayer(numInputs); // Special handling for input layer
        for (int layerIdx = 1; it != layerSizes.end(); ++it, ++layerIdx) {
            bool isOutputLayer = (std::next(it) == layerSizes.end());
            addLayer(layerIdx, *it, numInputs, isOutputLayer);
            numInputs = *it;
        }
    }

    void addInputLayer(int numNodes) {
        layers.emplace_back(0, numNodes, 1, true); // Each input node has 1 input
    }

    void addLayer(int layerIdx, int numNodes, int numInputsPerNode, bool isOutputLayer) {
        layers.emplace_back(layerIdx, numNodes, numInputsPerNode, false, isOutputLayer); // Pass numInputsPerNode to the layer constructor
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

    // Stream insertion and extraction operators for serialization
    friend std::ostream& operator<<(std::ostream& os, const Model& model) {
        os << model.layers.size() << " ";
        for (const auto& layer : model.layers) {
            os << layer << " ";
        }
        return os;
    }

    friend std::istream& operator>>(std::istream& is, Model& model) {
        size_t size;
        is >> size;
        model.layers.resize(size);
        for (auto& layer : model.layers) {
            is >> layer;
        }
        return is;
    }

    // Method for detailed state reporting
    std::string debugState() const {
        std::stringstream ss;
        ss << "Model state:\n";
        for (const auto& layer : layers) {
            ss << layer.debugState() << "\n";
        }
        return ss.str();
    }

private:
    std::vector<Layer<T, Activation>> layers;
};

#endif // MODEL_H
