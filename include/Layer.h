#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <iostream>
#include "Node.h"

template <typename T, ActivationType Activation>
class Layer {
public:
    Layer(int layerIdx, int numNodes, int numInputNodes, bool isInputLayer, bool isOutputLayer = false) {
        if (isInputLayer) {
            for (int i = 0; i < numNodes; ++i) {
                std::cout << "Node() input node " << i << std::endl;
                nodes.emplace_back(layerIdx, i, true); // Use constructor for input nodes
            }
        } else if (isOutputLayer) {
            for (int i = 0; i < numNodes; ++i) {
                std::cout << "Node(" << numInputNodes << ") output node " << i << std::endl;
                nodes.emplace_back(layerIdx, i, numInputNodes, true); // Use constructor for output nodes
            }
        } else {
            for (int i = 0; i < numNodes; ++i) {
                std::cout << "Node(" << numInputNodes << ") hidden node " << i << std::endl;
                nodes.emplace_back(layerIdx, i, numInputNodes); // Use constructor for hidden nodes
            }
        }
    }

    std::vector<T> forward(const std::vector<T>& inputs) const {
        std::vector<T> outputs;
        for (const auto& node : nodes) {
            outputs.push_back(node.activate(inputs));
        }
        return outputs;
    }

    const std::vector<Node<T, Activation>>& getNodes() const {
        return nodes;
    }

    // Stream insertion and extraction operators for serialization
    friend std::ostream& operator<<(std::ostream& os, const Layer& layer) {
        os << layer.nodes.size() << " ";
        for (const auto& node : layer.nodes) {
            os << node << " ";
        }
        return os;
    }

    friend std::istream& operator>>(std::istream& is, Layer& layer) {
        size_t size;
        is >> size;
        layer.nodes.resize(size);
        for (auto& node : layer.nodes) {
            is >> node;
        }
        return is;
    }

    // Method for detailed state reporting
    std::string debugState() const {
        std::stringstream ss;
        ss << "Layer state:\n";
        for (const auto& node : nodes) {
            ss << node.debugState() << "\n";
        }
        return ss.str();
    }

private:
    std::vector<Node<T, Activation>> nodes;
};

#endif // LAYER_H
