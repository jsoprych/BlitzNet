#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include "Node.h"

template <typename T, ActivationType Activation>
class Layer {
public:
    Layer(int numNodes, int numInputNodes, bool isInputLayer) {
        if (isInputLayer) {
            for (int i = 0; i < numNodes; ++i) {
                std::cout << "Node() input node " << i << std::endl;
                nodes.emplace_back(); // Use default constructor for input nodes
            }
        } else {
            for (int i = 0; i < numNodes; ++i) {
                std::cout << "Node(" << numInputNodes << ") layer node " << i << std::endl;
                nodes.emplace_back(numInputNodes); // Use the constructor for regular nodes with specified number of inputs
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

private:
    std::vector<Node<T, Activation>> nodes;
};

#endif // LAYER_H
