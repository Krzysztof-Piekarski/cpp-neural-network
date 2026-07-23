#include <cnn/Activation.h>
#include <cmath>

namespace cnn::activation{

double sigmoid(double x) noexcept{
    if(x >= 0.0){
        double e = std::exp(-x);
        return 1.0 / (1.0 + e);
    }
    double e = std::exp(x);
    return e / (1.0 + e);
}

double sigmoidDerivative(double x) noexcept{
    double sig = sigmoid(x);
    return sig * (1.0 - sig);
}

double relu(double x) noexcept{
    return x > 0.0 ? x : 0.0;
}

double reluDerivative(double x) noexcept{
    return x > 0.0 ? 1.0 : 0.0;
}

double tanh(double x) noexcept{
    return std::tanh(x);
}

double tanhDerivative(double x) noexcept{
    double tanh = std::tanh(x);
    return 1.0 - (tanh * tanh);
}

} // namespace cnn::activation