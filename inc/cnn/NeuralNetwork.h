#pragma once

#include <cnn/Matrix.h>
#include <cnn/DenseLayer.h>

namespace cnn{

class NeuralNetwork{
public:
    void addLayer(DenseLayer layer);

    Matrix predict(const Matrix& input) const;

private:
    std::vector<DenseLayer> layers_;
};

} // namespace cnn