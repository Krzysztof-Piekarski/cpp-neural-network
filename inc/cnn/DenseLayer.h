#pragma once

#include <cnn/Activation.h>
#include <cnn/Matrix.h>
#include <cnn/Types.h>

namespace cnn{

class DenseLayer{
    public:
        DenseLayer(size_t inputSize,
                   size_t outputSize,
                   ActivationFunction activation = activation::sigmoid);

        Matrix forward(const Matrix& input) const;

        const Matrix& weights() const noexcept;
        const Matrix& bias() const noexcept;

        void setWeights(Matrix);
        void setBias(Matrix);

        size_t inputSize() const noexcept;
        size_t outputSize() const noexcept;
    
    private:
        static Matrix createRandomWeights(size_t inputSize, size_t outputSize);

        Matrix weights_;
        Matrix bias_;
        ActivationFunction activation_;
};

} // namespace cnn