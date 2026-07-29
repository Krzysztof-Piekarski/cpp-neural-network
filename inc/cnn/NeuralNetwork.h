#pragma once

#include <cnn/Matrix.h>
#include <cnn/DenseLayer.h>

namespace cnn{

class NeuralNetwork{
public:
    void addLayer(DenseLayer layer);

    Matrix forwardTraining(const Matrix& input);
    void backward(const Matrix& lossGradient);
    void updateParameters(double learnigRate);

    Matrix predict(const Matrix& input) const;

private:
    std::vector<DenseLayer> layers_;
};

} // namespace cnn