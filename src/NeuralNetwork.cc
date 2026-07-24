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

} // namespace cnns