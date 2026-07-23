#include <cnn/Loss.h>
#include <stdexcept>

namespace cnn::loss{

double meanSquaredError(const Matrix& prediction,
                        const Matrix& target){
    if(prediction.rows() != target.rows() ||
       prediction.cols() != target.cols()){
        throw std::invalid_argument(
            "Prediction and target dimensions must match."
        );
    }

    double sum{0.0};

    for(size_t r{0}; r<prediction.rows(); ++r){
        for(size_t c{0}; c<prediction.cols(); ++c){
            double diff = prediction(r,c) - target(r,c);
            sum += diff * diff;
        }
    }

    return sum / prediction.size();
}

Matrix meanSquaredErrorDerivative(const Matrix& prediction,
                                  const Matrix& target){
    if(prediction.rows() != target.rows() ||
       prediction.cols() != target.cols()){
        throw std::invalid_argument(
            "Prediction and target dimensions must match."
        );
    }

    cnn::Matrix result(prediction.rows(), prediction.cols());
    const double scale = 2.0 / prediction.size();

    for(size_t r{0}; r<result.rows(); ++r){
        for(size_t c{0}; c<result.cols(); ++c){
            result(r,c) = scale * (prediction(r,c) - target(r,c));
        }
    }

    return result;
}

} // namespace cnn::loss