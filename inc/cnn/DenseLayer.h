#include <cnn/Activation.h>
#include <cnn/Matrix.h>
#include <functional>

namespace cnn{

class DenseLayer{
    public:
        using ActivationFunction = std::function<double(double)>;

        DenseLayer(size_t inputSize,
                   size_t outputSize,
                   ActivationFunction activation = activation::sigmoid);

        Matrix forward(const Matrix& input) const;

        const Matrix& weights() const noexcept;
        const Matrix& bias() const noexcept;
    
    private:
        Matrix weights_;
        Matrix bias_;
        ActivationFunction activation_;
};

} // namespace cnn