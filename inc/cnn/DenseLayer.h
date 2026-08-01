#pragma once

#include <cnn/Activation.h>
#include <cnn/Matrix.h>
#include <cnn/Types.h>
#include <optional>

namespace cnn{

class DenseLayer{
    public:
        DenseLayer(size_t inputSize,
                   size_t outputSize,
                   ActivationType activation = ActivationType::Sigmoid);

        Matrix forward(const Matrix& input) const;
        Matrix forwardTraining(const Matrix& input);
        Matrix backward(const Matrix& gradient);
        Matrix calculateOutputDelta(const Matrix& lossGradient) const;
        void updateParameters(double learningRate);

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
        ActivationType activation_;

        std::optional<Matrix> inputCache_;
        std::optional<Matrix> zCache_;
        std::optional<Matrix> outputCache_;

        std::optional<Matrix> weightsGradient_; 
        std::optional<Matrix> biasGradient_;
};

} // namespace cnn