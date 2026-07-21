#include <cnn/DenseLayer.h>
#include <cnn/Initializer.h>
#include <stdexcept>

namespace cnn{

DenseLayer::DenseLayer(size_t inputSize,
                       size_t outputSize,
                       ActivationFunction activation)
    : weights_(initializer::random(outputSize, inputSize)),
      bias_(initializer::zeros(outputSize, 1)),
      activation_(activation){}

Matrix DenseLayer::forward(const Matrix& input) const{
    Matrix output = weights_ * input;

    output += bias_;

    return output.map(activation_);
};

void DenseLayer::setWeights(const Matrix& weights){
    if(weights.rows() != weights_.rows() ||
       weights.cols() != weights_.cols()){
        throw std::invalid_argument("Invalid weights dimensions.");
       }
    weights_ = weights;
}

void DenseLayer::setBias(const Matrix& bias){
    if(bias.rows() != bias_.rows() ||
       bias.cols() != bias_.cols()){
        throw std::invalid_argument("Invalid bias dimensions.");
    }
    bias_ = bias;
}

} //namespace cnn