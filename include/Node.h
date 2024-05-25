#ifndef NODE_H
#define NODE_H

#include <vector>
#include <numeric>
#include <cassert>
#include <random>
#include <iostream>
#include <sstream>
#include "ActivationFunction.h"

template <typename T, ActivationType Activation>
class Node {
public:
    // Constructor for Input Layer Nodes
    Node(int layerIdx, int nodeIdx, bool isInputNode = true)
        : layerIdx(layerIdx), nodeIdx(nodeIdx), weights(1), bias(0), output(0), isInput(isInputNode), isOutput(false) {}

    // Constructor for Hidden/Output Layer Nodes
    Node(int layerIdx, int nodeIdx, int numInputs, bool isOutputNode = false)
        : layerIdx(layerIdx), nodeIdx(nodeIdx), weights(numInputs), bias(0), output(0), isInput(false), isOutput(isOutputNode) {
        assert(numInputs > 0 && "Number of inputs must be positive");
        initializeWeights();
    }

    // Template Metaprogramming (compile-time selectable) activation function
    T activate(const std::vector<T>& inputs) const {
        if (isInputNode()) {
            output = inputs[0];
        } else {
            T sum = std::inner_product(inputs.begin(), inputs.end(), weights.begin(), bias);
            output = ActivationFunction<T, Activation>::apply(sum);
        }
        std::cout << output << ": " << (*this) << std::endl;
        return output;
    }

    bool isInputNode() const {
        return isInput;
    }

    bool isOutputNode() const {
        return isOutput;
    }

    // Getter functions
    const std::vector<T>& getWeights() const {
        return weights;
    }

    T getBias() const {
        return bias;
    }

    T getOutput() const {
        return output;
    }

    int getLayerIdx() const {
        return layerIdx;
    }

    int getNodeIdx() const {
        return nodeIdx;
    }

    // Stream insertion and extraction operators for serialization
    friend std::ostream& operator<<(std::ostream& os, const Node& node) {
        os << "n" << node.layerIdx << ":" << node.nodeIdx;
        if (node.isInputNode()) {
            os << "i";
        } else if (node.isOutputNode()) {
            os << "o";
        }
        os << " " << node.weights.size() << " ";
        for (const auto& weight : node.weights) {
            os << weight << " ";
        }
        os << node.bias << " " << node.output;
        return os;
    }

    friend std::istream& operator>>(std::istream& is, Node& node) {
        std::string nodeId;
        size_t size;
        is >> nodeId >> size;
        node.weights.resize(size);
        for (auto& weight : node.weights) {
            is >> weight;
        }
        is >> node.bias >> node.output;

        // Parse layerIdx, nodeIdx, and type from the combined string
        size_t colonPos = nodeId.find(':');
        node.layerIdx = std::stoi(nodeId.substr(1, colonPos - 1));
        node.nodeIdx = std::stoi(nodeId.substr(colonPos + 1, nodeId.size() - colonPos - 2));

        char nodeType = nodeId.back();
        node.isInput = (nodeType == 'i');
        node.isOutput = (nodeType == 'o');

        return is;
    }

    // Method for detailed state reporting
    std::string debugState() const {
        std::stringstream ss;
        ss << "Node [Layer: " << layerIdx << ", Position: " << nodeIdx << "] - Weights: ";
        for (const auto& weight : weights) {
            ss << weight << " ";
        }
        ss << "Bias: " << bias << " Output: " << output;
        return ss.str();
    }

private:
    int layerIdx;                   // Index of the layer the node belongs to
    int nodeIdx;                    // Position index of the node within the layer
    mutable std::vector<T> weights; // Allow modification in const member function
    mutable T bias;                 // Allow modification in const member function
    mutable T output;               // Allow modification in const member function
    bool isInput;                   // True if node is an input node, false otherwise
    bool isOutput;                  // True if node is an output node, false otherwise

    void initializeWeights() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1.0, 1.0);
        for (auto& weight : weights) {
            weight = dis(gen);
        }
    }
};

#endif // NODE_H
