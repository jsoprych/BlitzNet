#ifndef NODE_H
#define NODE_H

#include <vector>
#include <numeric>
#include <cassert>
#include <random>
#include <iostream>
#include "ActivationFunction.h"

template <typename T, ActivationType Activation>
class Node {
public:
    // Constructor for Input Layer Nodes
    Node() : weights(1), bias(0), output(0), isInput(true) {}

    // Constructor for Hidden/Output Layer Nodes
    Node(int numInputs) : weights(numInputs), bias(0), output(0), isInput(false){
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

    // Stream insertion and extraction operators for serialization
    friend std::ostream& operator<<(std::ostream& os, const Node& node) {
        os << node.weights.size() << " ";
        for (const auto& weight : node.weights) {
            os << weight << " ";
        }
        os << node.bias << " " << node.output;
        return os;
    }

    friend std::istream& operator>>(std::istream& is, Node& node) {
        size_t size;
        is >> size;
        node.weights.resize(size);
        for (auto& weight : node.weights) {
            is >> weight;
        }
        is >> node.bias >> node.output;
        return is;
    }

private:
    mutable std::vector<T> weights; // Allow modification in const member function
    mutable T bias;                 // Allow modification in const member function
    mutable T output;               // Allow modification in const member function
    bool isInput;                   // True if node is an input node, false otherwise

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
