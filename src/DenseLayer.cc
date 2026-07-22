#include <cnn/DenseLayer.h>
#include <cnn/Initializer.h>
#include <stdexcept>
#include <utility>

namespace cnn{

DenseLayer::DenseLayer(size_t inputSize,
                       size_t outputSize,
                       ActivationFunction activation)
    : weights_(createRandomWeights(inputSize, outputSize)),
      bias_(initializer::zeros(outputSize, 1)),
      activation_(std::move(activation)){}

Matrix DenseLayer::forward(const Matrix& input) const{
    if(input.rows() != weights_.cols()){
        throw std::invalid_argument("Invalid input dimensions.");
    }

    Matrix output = weights_ * input;
    output += bias_;

    return output.map(activation_);
}

Matrix DenseLayer::createRandomWeights(size_t inputSize, size_t outputSize){
    if(inputSize == 0 || outputSize == 0){
        throw std::invalid_argument(
            "Layer dimensions must be greater than zero.");
    }
    
    return initializer::random(outputSize, inputSize);
}


const Matrix& DenseLayer::weights() const noexcept{
    return weights_;
}

const Matrix& DenseLayer::bias() const noexcept{
    return bias_;
}

void DenseLayer::setWeights(Matrix weights){
    if(weights.rows() != weights_.rows() ||
       weights.cols() != weights_.cols()){
        throw std::invalid_argument("Invalid weights dimensions.");
       }
    weights_ = std::move(weights);
}

void DenseLayer::setBias(Matrix bias){
    if(bias.rows() != bias_.rows() ||
       bias.cols() != bias_.cols()){
        throw std::invalid_argument("Invalid bias dimensions.");
    }
    bias_ = std::move(bias);
}

} //namespace cnn