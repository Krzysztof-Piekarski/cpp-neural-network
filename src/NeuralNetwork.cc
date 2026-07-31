#include <cnn/NeuralNetwork.h>
#include <cnn/Loss.h>
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
    checkIfNotEmpty();

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
    checkIfNotEmpty();

    Matrix gradient = lossGradient;

    for(auto it = layers_.rbegin(); it != layers_.rend(); ++it){
        gradient = it->backward(gradient);
    }
}

void NeuralNetwork::updateParameters(double learnigRate){
    checkIfNotEmpty();

    for(auto& layer : layers_){
        layer.updateParameters(learnigRate);
    }
}

void NeuralNetwork::fit(const Matrix& input,
                        const Matrix& target,
                        double learningRate){
    checkIfNotEmpty();
    
    if(target.rows() != layers_.back().outputSize() ||
       target.cols() != 1){
        throw std::invalid_argument(
            "Target size does not match network output."
        );
    }

    if(learningRate <= 0.0){
        throw std::invalid_argument(
            "Learning rate must be greater than zero."
        );
    }

    Matrix prediction = forwardTraining(input);

    Matrix gradient = loss::meanSquaredErrorDerivative(prediction, target);

    backward(gradient);

    updateParameters(learningRate);
}

Matrix NeuralNetwork::predict(const Matrix& input) const{
    checkIfNotEmpty();

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

void NeuralNetwork::checkIfNotEmpty() const{
    if(layers_.empty()){
        throw std::logic_error(
            "Neural network has no layers."
        );
    }
}

} // namespace cnn