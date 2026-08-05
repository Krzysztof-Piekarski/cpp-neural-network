#include <stdio.h>
#include <cmath>
#include <numbers>
#include <cnn/NeuralNetwork.h>

int main(){
    std::vector<cnn::Matrix> inputs;
    std::vector<cnn::Matrix> targets;
    double learningRate{0.01};

    const double pi = std::numbers::pi;
    for(double x = -pi; x <= pi; x += 0.05){
        inputs.push_back({{x/pi}});
        targets.push_back({{std::sin(x)}});
    }

    cnn::NeuralNetwork net;
    cnn::DenseLayer layer1(1,16, cnn::ActivationType::Tanh);
    cnn::DenseLayer layer2(16,16, cnn::ActivationType::Tanh);
    cnn::DenseLayer layer3(16,1, cnn::ActivationType::Tanh);
    
    net.addLayer(layer1);
    net.addLayer(layer2);
    net.addLayer(layer3);

    for(int epoch{0}; epoch <= 5000; ++epoch){
        for(int i{0}; i < inputs.size(); i++){
            net.fit(inputs[i], targets[i], learningRate);
        }

        if(epoch%1000 == 0){
            printf("\nEpoch: %d\n", epoch);
            
            for(double x = -pi; x <= pi; x += pi/8.0){
                cnn::Matrix prediction = net.predict(cnn::Matrix{{x/pi}});
                printf("x=%7.3f   sin(x)=%8.5f   pred(x)=%8.5f\n", x, std::sin(x), prediction(0,0));
            }
        }
    }
}