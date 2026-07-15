#include <cnn/Activation.h>
#include <cmath>

namespace cnn::activation{

double sigmoid(double x) noexcept{
    if (x >= 0.0){
        double e = std::exp(-x);
        return 1.0 / (1.0 + e);
    }
    double e = std::exp(x);
    return e / (1.0 + e);
}

/*double relu(double);

double tanh(double);

double sigmoidDerivative(double);

double reluDerivative(double);*/

} // namespace cnn::activation