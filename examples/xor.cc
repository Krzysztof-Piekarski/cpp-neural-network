#include <stdio.h>
#include <cnn/NeuralNetwork.h>

int main(){
    std::vector<cnn::Matrix> inputs{
        {{0.0}, {0.0}},
        {{1.0}, {0.0}},
        {{0.0}, {1.0}},
        {{1.0}, {1.0}}
    };
    std::vector<cnn::Matrix> targets{
        {{0.0}},
        {{1.0}},
        {{1.0}},
        {{0.0}}
    };
    double learningRate{0.1};

    cnn::NeuralNetwork net;
    cnn::DenseLayer layer1(2,8, cnn::ActivationType::Relu);
    cnn::DenseLayer layer2(8,8, cnn::ActivationType::Relu);
    cnn::DenseLayer layer3(8,1, cnn::ActivationType::Sigmoid);
    
    net.addLayer(layer1);
    net.addLayer(layer2);
    net.addLayer(layer3);

    for(int epoch{0}; epoch <=5000; ++epoch){
        for(int i{0}; i < inputs.size(); i++){
            net.fit(inputs[i], targets[i], learningRate);
        }

        if(epoch%100 == 0){
            printf("\nEpoch: %d\n", epoch);
            
            for(int i{0}; i < inputs.size(); i++){
                printf("[%1.1f], [%1.1f] -> %1.9f\n",
                inputs[i].operator()(0,0),
                inputs[i].operator()(1,0),
                net.predict(inputs[i]).operator()(0,0));
            }
        }
    }
}