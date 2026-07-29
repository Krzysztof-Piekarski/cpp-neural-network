#include <cnn/NeuralNetwork.h>
#include <stdexcept>

namespace cnn{

void NeuralNetwork::addLayer(DenseLayer layer){
    if(!layers_.empty()){
        if(layers_.back().outputSize() != layer.inputSize()){
            throw std::invalid_argument(
                "Layer input size does not match previous layer output size."
            );
        }
    }

    layers_.push_back(std::move(layer));
}

Matrix NeuralNetwork::forwardTraining(const Matrix& input){
    if(layers_.empty()){
        throw std::logic_error(
            "Neural network has no layers."
        );
    }

    if(input.rows() != layers_.front().inputSize()){
        throw std::invalid_argument(
            "Input size does not match first layer."
        );
    }

    Matrix output = input;

    for(auto& layer : layers_){
        output = layer.forwardTraining(output);
    }

    return output;
}

void NeuralNetwork::backward(const Matrix& lossGradient){
    if(layers_.empty()){
        throw std::logic_error(
            "Neural network has no layers."
        );
    }

    Matrix gradient = lossGradient;

    for(auto it = layers_.rbegin(); it != layers_.rend(); ++it){
        gradient = it->backward(gradient);
    }
}

void NeuralNetwork::updateParameters(double learnigRate){
    if(layers_.empty()){
        throw std::logic_error(
            "Neural network has no layers."
        );
    }

    for(auto& layer : layers_){
        layer.updateParameters(learnigRate);
    }
}

Matrix NeuralNetwork::predict(const Matrix& input) const{
    if(layers_.empty()){
        throw std::logic_error(
            "Neural network has no layers."
        );
    }

    if(input.rows() != layers_.front().inputSize()){
        throw std::invalid_argument(
            "Input size does not match first layer."
        );
    }

    Matrix output = input;

    for(const auto& layer : layers_){
        output = layer.forward(output);
    }

    return output;
}

} // namespace cnn