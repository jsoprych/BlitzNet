#ifndef LOSSFUNCTION_H
#define LOSSFUNCTION_H

#include <vector>
#include <cmath>

template <typename T>
class LossFunction {
public:
    static T meanSquaredError(const std::vector<T>& outputs, const std::vector<T>& targets) {
        if (outputs.size() != targets.size()) {
            throw std::invalid_argument("Size mismatch between outputs and targets.");
        }
        T mse = 0;
        for (size_t i = 0; i < outputs.size(); ++i) {
            mse += (outputs[i] - targets[i]) * (outputs[i] - targets[i]);
        }
        return mse / outputs.size();
    }
};

#endif // LOSSFUNCTION_H
