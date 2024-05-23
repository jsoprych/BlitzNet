#ifndef ACTIVATIONFUNCTION_H
#define ACTIVATIONFUNCTION_H

#include <cmath>

enum class ActivationType {
    None,
    Sigmoid
};

template <typename T, ActivationType Activation>
class ActivationFunction;

template <typename T>
class ActivationFunction<T, ActivationType::None> {
public:
    static T apply(T input) {
        return input;
    }
};

template <typename T>
class ActivationFunction<T, ActivationType::Sigmoid> {
public:
    static T apply(T input) {
        return 1 / (1 + std::exp(-input));
    }
};

#endif // ACTIVATIONFUNCTION_H
